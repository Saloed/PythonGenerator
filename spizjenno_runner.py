import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from spizjenno.analyser_rnn import sequence_input, labels_output
from spizjenno.attention_dynamic_rnn import Attention
from spizjenno.dynamic_rnn import bidirectional_dynamic_rnn


class MultiAttention(Attention):
    def call(self, state):
        cur_state = state[:, -self._state_size:]
        return super(MultiAttention, self).call(cur_state)
        # state[:, -self._state_size:] = state_with_attention
        # other_states = state[:, :-self._state_size]
        # new_state = tf.concat((other_states, state_with_attention), 1)
        # return new_state


def run(input_num_tokens, num_labels, batch_size):
    target_labels = tf.placeholder(tf.int32, [None, batch_size])
    target_length = tf.placeholder(tf.int32, [batch_size])
    with tf.variable_scope("Input"):
        input_ids = tf.placeholder(tf.int32, [None, batch_size])
        inputs_length = tf.placeholder(tf.int32, [batch_size])
        inputs = tf.one_hot(input_ids, input_num_tokens)
        inputs = tf.transpose(inputs, [1, 0, 2])
    with tf.variable_scope("Analyser", dtype=tf.float32) as scope:
        dtype = scope.dtype
        cell_fw = tf_rnn.GRUCell(128)
        cell_bw = tf_rnn.GRUCell(128)
        _, attention_states = bidirectional_dynamic_rnn(
            cell_fw, cell_bw, inputs, inputs_length, dtype=dtype, time_major=False
        )
        attention_states = tf.concat(attention_states, 2)
        num_cells = 3
        decoder_state_size = 256
        labels_attention = MultiAttention(attention_states, decoder_state_size, dtype=dtype, scope="LabelsAttention")
        cells = [tf_rnn.GRUCell(decoder_state_size) for _ in range(num_cells)]
        labels_cell = tf_rnn.MultiRNNCell(cells, state_is_tuple=False)
        labels_logits, raw_labels, labels_states, attentions, weights = labels_output(
            labels_cell, labels_attention, num_labels, tf.reduce_max(target_length),
            hidden_size=256, dtype=dtype, time_major=True,
        )

    with tf.variable_scope("Loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=labels_logits,
            labels=target_labels,
        )
        loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(target_length))
        outputs = tf.nn.softmax(labels_logits)
        outputs = tf.argmax(outputs, axis=-1)

    return input_ids, inputs_length, target_labels, target_length, outputs, loss
