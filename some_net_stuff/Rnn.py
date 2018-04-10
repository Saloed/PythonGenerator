import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell, AttentionCellWrapper

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

_Linear = core_rnn_cell._Linear


class GRUCellWithAttention(GRUCell):
    def __init__(
            self,
            num_units,
            attention_states,
            activation=None,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None,
            name=None
    ):
        super().__init__(num_units, activation, reuse, kernel_initializer, bias_initializer, name)
        self._attn_states = attention_states
        self._query_to_attention = None
        self._attention_to_query = None

    def call(self, inputs, state):
        cell_output, new_state = super().call(inputs, state)
        new_state = self._attention(new_state)
        return cell_output, new_state

    def _attention(self, query):
        conv2d = nn_ops.conv2d
        reduce_sum = math_ops.reduce_sum
        softmax = nn_ops.softmax
        tanh = math_ops.tanh

        with vs.variable_scope("attention"):
            attn_shape = self._attn_states.get_shape()
            batch_size = attn_shape[1]
            attn_vec_size = attn_shape[2]
            query_shape = query.get_shape()
            query_batch_size = query_shape[0]
            query_size = query_shape[1]
            assert query_batch_size == batch_size, 'Batch size must be equal'
            hidden = array_ops.reshape(
                tensor=self._attn_states,
                shape=[batch_size, -1, 1, attn_vec_size]
            )

            k = vs.get_variable("attn_w", [1, 1, attn_vec_size, attn_vec_size])
            v = vs.get_variable("attn_v", [attn_vec_size])
            hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            if self._query_to_attention is None:
                with vs.variable_scope("query2attention"):
                    self._query_to_attention = _Linear(query, attn_vec_size, True)
            y = self._query_to_attention(query)
            y = array_ops.reshape(y, [batch_size, 1, 1, attn_vec_size])
            s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
            a = softmax(s)
            a = array_ops.reshape(a, [batch_size, -1, 1, 1])
            d = a * hidden
            new_attns = reduce_sum(d, [1, 2])
            if self._attention_to_query is None:
                with vs.variable_scope("attention2query"):
                    self._attention_to_query = _Linear(new_attns, query_size, True)
            new_attns = self._attention_to_query(new_attns)
            return new_attns


def dynamic_rnn_with_states(cell, max_time_steps, sequence_end_marker, initial_state, initial_input):
    time = array_ops.constant(0, dtype=tf.int32, name="time")
    output = initial_input
    state = initial_state

    def _time_step(_time, _output, _output_ta, _state, _state_ta):
        _output, _state = cell(_output, _state)
        _output_ta = _output_ta.write(_time, _output)
        _state_ta = _state_ta.write(_time, _state)
        return _time + 1, _output, _output_ta, _state, _state_ta

    output_ta = tf.TensorArray(
        dtype=tf.float32,
        size=0,
        dynamic_size=True,
        element_shape=output.shape,
        tensor_array_name='output_ta'
    )

    state_ta = tf.TensorArray(
        dtype=tf.float32,
        size=0,
        dynamic_size=True,
        element_shape=state.shape,
        tensor_array_name='state_ta'
    )

    _, final_output, final_output_ta, final_state, final_state_ta = control_flow_ops.while_loop(
        cond=lambda time, output, *_: tf.logical_and(time < max_time_steps, output != sequence_end_marker),
        body=_time_step,
        loop_vars=(time, output, output_ta, state, state_ta),
        parallel_iterations=1,
        maximum_iterations=max_time_steps,
    )

    return final_output, final_output_ta, final_state, final_state_ta
