import tensorflow as tf


class MultiRnnWithDropout(tf.nn.rnn_cell.DropoutWrapper):

    def __init__(self, num_layers, state_size, dropout_prob):
        internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(num_layers)]
        internal_cell = tf.nn.rnn_cell.MultiRNNCell(internal_cells)
        super(MultiRnnWithDropout, self).__init__(
            cell=internal_cell,
            output_keep_prob=dropout_prob,
            state_keep_prob=dropout_prob
        )


class BasicMultiRNNCell(MultiRnnWithDropout):

    def __init__(self, num_layers, state_size, output_size, batch_size, dropout_prob):
        self._initial_state = None
        self._output_size = output_size
        self.batch_size = batch_size
        super(BasicMultiRNNCell, self).__init__(num_layers, state_size, dropout_prob)

    @property
    def output_size(self):
        return self._output_size

    @property
    def last_state_size(self):
        return self.state_size[-1]

    def initialize_outputs_ta(self, maximum_iterations):
        return tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[self.batch_size, self.output_size]
        )

    def set_initial_state(self, initial_state):
        self._initial_state = initial_state

    def get_initial_state(self):
        if self._initial_state is None:
            raise Exception('Initial state not set')
        initial_state = self.zero_state(self.batch_size, tf.float32)
        initial_state = list(initial_state)
        initial_state[0] = self._initial_state
        initial_state = tuple(initial_state)
        return initial_state

    def get_initial_input(self):
        return tf.zeros([self.batch_size, self.output_size], dtype=tf.float32)

    def finalize_outputs(self, final_outputs_ta):
        return final_outputs_ta.stack()

    def write_outputs(self, time, outputs_ta, outputs):
        outputs_ta = outputs_ta.write(time, outputs)
        return outputs_ta
