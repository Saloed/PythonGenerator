import tensorflow as tf
from tensorflow import variable_scope as vs

from basic_rnn_cell import BasicMultiRNNCell


def compute_alphas(h_t, h_s):
    _score = tf.matmul(h_t, h_s, transpose_b=True)
    score = tf.squeeze(_score, [1])
    alpha_ts = tf.nn.softmax(score)
    alpha_ts = tf.expand_dims(alpha_ts, 2)
    return alpha_ts


def attention(
        hidden_state,
        source_states,
        attention_vec_size,
        scope=None
):
    with vs(scope or "attention") as varscope:
        h_s = source_states
        h_t = tf.expand_dims(hidden_state, 1)
        alpha_ts = compute_alphas(h_t, h_s)
        weighted_sources = alpha_ts * h_s
        context = tf.reduce_sum(weighted_sources, axis=1)
        combined = tf.concat([context, hidden_state], axis=1)
        combined_shape = combined.shape[1]
        W_c = tf.get_variable('W_c', shape=[combined_shape, attention_vec_size], dtype=tf.float32)
        multiplied = combined @ W_c
        attention_vec = tf.tanh(multiplied)
        return attention_vec


def get_attention(attention_source, attention_vec_size):
    def apply_attention(cell, inputs, state, scope):
        attention_vec = attention(inputs, attention_source, attention_vec_size, scope)
        return attention_vec, state

    return apply_attention


def get_output_projection(output_size):
    def apply_projection(cell, inputs, state, scope):
        output_projection_w = tf.get_variable(
            name='output_projection_w',
            shape=[cell.last_state_size, output_size],
            dtype=tf.float32,
        )
        output_projection_b = tf.get_variable(
            name='output_projection_b',
            shape=[output_size],
            dtype=tf.float32,
        )

        projected_output = tf.nn.relu_layer(inputs, output_projection_w, output_projection_b)
        return projected_output, state

    return apply_projection


class RnnCellProxy:

    def __init__(self, cell, pre_functions=None, post_functions=None):
        self.__proxy__data__ = {
            'cell': cell,
            'pre_fun': pre_functions or [],
            'post_fun': post_functions or [],
        }

    def __call__(self, inputs, state, scope=None):
        cell = self.__proxy__data__['cell']
        pre_fun = self.__proxy__data__['pre_fun']
        post_fun = self.__proxy__data__['post_fun']
        for fun in pre_fun:
            inputs, state = fun(cell, inputs, state, scope)
        inputs, state = cell(inputs, state, scope)
        for fun in post_fun:
            inputs, state = fun(cell, inputs, state, scope)
        return inputs, state

    def __getattr__(self, item):
        if item == '__proxy__data__':
            return object.__getattribute__(self, '__proxy__data__')
        return getattr(self.__proxy__data__['cell'], item)

    def __setattr__(self, key, value):
        if key == '__proxy__data__':
            return object.__setattr__(self, '__proxy__data__', value)
        setattr(self.__proxy__data__['cell'], key, value)


def get_rnn_cell(
        num_layers, state_size, output_size, batch_size, dropout_prob,
        output_projection=False, projection_size=None,
        attention=False, attention_source=None, attention_vec_size=None
):
    basic_cell = BasicMultiRNNCell(num_layers, state_size, output_size, batch_size, dropout_prob)

    post_functions = []
    if attention and attention_source is not None and attention_vec_size is not None:
        post_functions.append(get_attention(attention_source, attention_vec_size))
    if output_projection and projection_size is not None:
        post_functions.append(get_output_projection(projection_size))

    proxy_cell = RnnCellProxy(basic_cell, post_functions=post_functions)
    return proxy_cell