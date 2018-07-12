import collections

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util

from basic_rnn_cell import BasicMultiRNNCell


class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("cell_state", "last_ids", "prob_c"))
):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return tf.nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))


class CopyMultiRnnCell(BasicMultiRNNCell):

    def get_initial_state(self):
        initial_state = super(CopyMultiRnnCell, self).get_initial_state()
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[self.batch_size]):
            last_ids = tf.zeros([self.batch_size], tf.int32) - 1
            prob_c = tf.zeros([self.batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=initial_state, last_ids=last_ids, prob_c=prob_c)

    def initialize_outputs_ta(self, maximum_iterations):
        outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[self.batch_size, self.output_size]
        )
        copy_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
        )
        return outputs_ta, copy_ta

    def get_initial_input(self):
        inputs = tf.zeros([self.batch_size, self.output_size], dtype=tf.float32)
        last_copy = tf.zeros([self.batch_size, self._encoder_seq_length], dtype=tf.float32)
        return inputs, last_copy

    def finalize_outputs(self, final_outputs_ta):
        outputs_ta, copy_ta = final_outputs_ta
        outputs_ta = outputs_ta.stack()
        copy_ta = copy_ta.stack()
        return outputs_ta, copy_ta

    def write_outputs(self, time, outputs_ta, outputs):
        outputs_ta, copy_ta = outputs_ta
        outputs, copy = outputs
        outputs_ta = outputs_ta.write(time, outputs)
        copy_ta = copy_ta.write(time, copy)
        return outputs_ta, copy_ta

    def __init__(self,
                 num_layers, state_size, output_size, batch_size, dropout_prob,
                 encoder_states, encoder_seq_length, encoder_state_size,
                 encoder_input_ids, gen_vocab_size=None):
        super(CopyMultiRnnCell, self).__init__(num_layers, state_size, output_size, batch_size, dropout_prob)

        self._gen_vocab_size = gen_vocab_size or self.output_size

        self._encoder_seq_length = encoder_seq_length

        self._encoder_input_ids = tf.transpose(encoder_input_ids, [1, 0])
        self._encoder_states = tf.transpose(encoder_states, [1, 0, 2])
        self._encoder_state_size = encoder_state_size

        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size, self.last_state_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                            "Received type %s instead." % type(state))

        inputs, last_copy = inputs

        last_ids = state.last_ids
        prob_c = state.prob_c
        cell_state = state.cell_state

        mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1), self._encoder_input_ids), tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1)
        condition = tf.less(mask_sum, 1e-7)
        normalized_mask = mask / tf.expand_dims(mask_sum, 1)

        # condition = utilss.print_shape(condition, 'condition')
        # mask = utilss.print_shape(mask, 'mask')
        # normalized_mask = utilss.print_shape(normalized_mask, 'norm mask')
        # prob_c = utilss.print_shape(prob_c, 'prob c')

        mask = tf.where(condition, mask, normalized_mask, name='select_mask')
        rou = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou, name='einsum_selective_input')
        inputs = tf.concat([inputs, selective_read], 1)

        outputs, cell_state = super(CopyMultiRnnCell, self).__call__(inputs, cell_state, scope)

        generate_score = self._projection(outputs)

        copy_score = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight, name='einsum_copy_score')
        copy_score = tf.nn.tanh(copy_score)

        copy_score = tf.einsum("ijm,im->ij", copy_score, outputs, name='einsum_copy_score_2')

        expand_mask = tf.cast(tf.greater_equal(self._encoder_input_ids, 0), tf.float32)
        expanded_copy_score = expand_mask * copy_score

        prob_g = generate_score
        prob_c = expanded_copy_score

        #        mixed_score = tf.concat([generate_score, expanded_copy_score], 1)
        #        probs = tf.nn.softmax(mixed_score)
        #        prob_g = probs[:, :self._gen_vocab_size]
        #        prob_c = probs[:, self._gen_vocab_size:]

        encoder_input_mask = tf.one_hot(self._encoder_input_ids, self.output_size)
        prob_c_one_hot = tf.einsum("ijn,ij->in", encoder_input_mask, prob_c, name='einsum_prob_c_one_hot')
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self.output_size - self._gen_vocab_size]])
        outputs = prob_c_one_hot + prob_g_total
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        # prob_c.set_shape([None, self._encoder_state_size])
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)

        return (outputs, copy_score), state


    @property
    def last_state_size(self):
        return super(CopyMultiRnnCell, self).state_size[-1]

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self.last_state_size, last_ids=tf.TensorShape([]),
                                   prob_c=self._encoder_state_size)
