import tensorflow as tf

from tensorflow import variable_scope as vs

import copy_net
from utilss import dict_to_object

from net_conf import *
# from new_res.model_1_conf import *

# from results.model_9_net_conf import *


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
            shape=[cell.output_size, output_size],
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
        num_layers, state_size, dropout_prob,
        output_projection=False, projection_size=None,
        attention=False, attention_source=None, attention_vec_size=None
):
    internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(num_layers)]
    basic_cell = tf.nn.rnn_cell.MultiRNNCell(internal_cells)
    basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=dropout_prob, state_keep_prob=dropout_prob)
    post_functions = []
    if attention and attention_source is not None and attention_vec_size is not None:
        post_functions.append(get_attention(attention_source, attention_vec_size))
    if output_projection and projection_size is not None:
        post_functions.append(get_output_projection(projection_size))

    proxy_cell = RnnCellProxy(basic_cell, post_functions=post_functions)
    return proxy_cell


def dynamic_decode(
        decoder_cell, batch_size, decoder_initial_input, output_size,
        decoder_initial_state,
        maximum_iterations=None, parallel_iterations=32, swap_memory=False, scope=None
):
    with vs(scope or "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations, dtype=tf.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_state = decoder_cell.zero_state(batch_size, tf.float32)

        if decoder_initial_state is not None:
            initial_state = list(initial_state)
            initial_state[0] = decoder_initial_state
            initial_state = tuple(initial_state)

        initial_inputs = decoder_initial_input

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[batch_size, output_size]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = decoder_cell(inputs, state)
            outputs_ta = outputs_ta.write(time, cell_output)
            return time + 1, outputs_ta, cell_state, cell_output

        _, final_outputs_ta, final_state, *_ = tf.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs = final_outputs_ta.stack()

    return final_outputs, final_state


def build_encoder(input_ids, input_length, input_num_tokens, encoder_input_size, state_size, layers, scope,
                  dropout_prob):
    embedding = tf.get_variable(
        name='input_embedding',
        shape=[input_num_tokens, encoder_input_size],
        dtype=tf.float32
    )

    prepared_inputs = tf.nn.embedding_lookup(embedding, input_ids)
    encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(layers)]
    encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(layers)]

    encoder_fw_internal_cells = [
        tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob, state_keep_prob=dropout_prob)
        for cell in encoder_fw_internal_cells
    ]
    encoder_bw_internal_cells = [
        tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob, state_keep_prob=dropout_prob)
        for cell in encoder_bw_internal_cells
    ]

    from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
    return stack_bidirectional_dynamic_rnn(
        cells_fw=encoder_fw_internal_cells,
        cells_bw=encoder_bw_internal_cells,
        inputs=prepared_inputs,
        sequence_length=input_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope,
    )


def build_model(
    batch_size, input_num_tokens, code_output_num_tokens, word_output_num_tokens
):
    with vs("inputs"):
        input_ids = tf.placeholder(tf.int32, [None, batch_size], 'input_ids')
        input_length = tf.placeholder(tf.int32, [batch_size], 'input_length')
        code_target_labels = tf.placeholder(tf.int32, [None, batch_size], 'code_outputs')
        code_target_length = tf.placeholder(tf.int32, [batch_size], 'output_length')
        word_target_labels = tf.placeholder(tf.int32, [None, batch_size], 'word_outputs')
        word_target_length = tf.placeholder(tf.int32, [batch_size], 'word_length')

        copyable_input_ids = tf.placeholder(tf.int32, [None, batch_size], 'copyable_input_ids')

        use_dropout = tf.placeholder(tf.bool, [], 'use_dropout')
        dropout_prob = tf.cond(use_dropout, lambda: DROPOUT_PROB, lambda: 1.0)

    with vs("encoder") as encoder_scope:
        encoder_output, encoder_state_fw, encoder_state_bw = build_encoder(
            input_ids, input_length, input_num_tokens, ENCODER_INPUT_SIZE, ENCODER_STATE_SIZE,
            ENCODER_LAYERS, encoder_scope, dropout_prob
        )

        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        # encoder_fw_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_fw_internal_cells, state_is_tuple=False)
        # encoder_bw_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_bw_internal_cells, state_is_tuple=False)
        #
        # encoder_output, (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        #     cell_fw=encoder_fw_cell,
        #     cell_bw=encoder_bw_cell,
        #     inputs=weighted_inputs,
        #     sequence_length=input_length,
        #     time_major=True,
        #     dtype=tf.float32,
        #     scope=encoder_scope,
        # )
        # final_encoder_state_fw = encoder_state_fw[:, -encoder_state_size:]
        # final_encoder_state_bw = encoder_state_bw[:, -encoder_state_size:]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

        # [batch_size x time x num_units]
        attention_source_states = tf.transpose(encoder_output, [1, 0, 2])

    with vs("code_decoder") as decoder_scope:
        decoder_state_size = ENCODER_STATE_SIZE * 2
        decoder_cell = get_rnn_cell(
            num_layers=CODE_DECODER_LAYERS,
            state_size=decoder_state_size,
            dropout_prob=dropout_prob,
            output_projection=True,
            projection_size=code_output_num_tokens,
            attention=CODE_DECODER_ATTENTION,
            attention_source=attention_source_states,
            attention_vec_size=decoder_state_size,
        )
        decoder_initial_input = tf.zeros([batch_size, code_output_num_tokens], dtype=tf.float32)
        code_decoder_output, code_decoder_state = dynamic_decode(
            decoder_cell=decoder_cell, batch_size=batch_size,
            decoder_initial_state=encoder_result,
            decoder_initial_input=decoder_initial_input,
            output_size=code_output_num_tokens,
            maximum_iterations=tf.reduce_max(code_target_length),
            parallel_iterations=1, scope=decoder_scope)

    with vs("word_decoder") as decoder_scope:
        decoder_state_size = ENCODER_STATE_SIZE * 2
        if WORD_COPY_NET:
            internal_cells = [tf.nn.rnn_cell.GRUCell(decoder_state_size) for _ in range(WORD_DECODER_LAYERS)]
            basic_cell = tf.nn.rnn_cell.MultiRNNCell(internal_cells)
            basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=dropout_prob,
                                                       state_keep_prob=dropout_prob)
            initial_state = basic_cell.zero_state(batch_size, tf.float32)
            initial_state = list(initial_state)
            initial_state[0] = encoder_result
            initial_state = tuple(initial_state)
            decoder_cell = copy_net.CopyNetWrapper(
                cell=basic_cell,
                encoder_states=encoder_output,
                encoder_input_ids=copyable_input_ids,
                vocab_size=word_output_num_tokens,
                initial_cell_state=initial_state
            )
            decoder_initial_state = None
        else:
            decoder_cell = get_rnn_cell(
                num_layers=WORD_DECODER_LAYERS,
                state_size=decoder_state_size,
                dropout_prob=dropout_prob,
                output_projection=True,
                projection_size=word_output_num_tokens,
                attention=WORD_DECODER_ATTENTION,
                attention_source=attention_source_states,
                attention_vec_size=decoder_state_size,
            )
            decoder_initial_state = encoder_result
        decoder_initial_input = tf.zeros([batch_size, word_output_num_tokens], dtype=tf.float32)
        word_decoder_output, word_decoder_state = dynamic_decode(
            decoder_cell=decoder_cell, batch_size=batch_size,
            decoder_initial_state=decoder_initial_state,
            decoder_initial_input=decoder_initial_input,
            output_size=word_output_num_tokens,
            maximum_iterations=tf.reduce_max(word_target_length),
            parallel_iterations=1, scope=decoder_scope)

    with vs("loss"):
        code_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=code_decoder_output,
            labels=code_target_labels,
        )

        # Calculate the average log perplexity
        code_loss = tf.reduce_sum(code_losses) / tf.to_float(tf.reduce_sum(code_target_length))

        word_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=word_decoder_output,
            labels=word_target_labels,
        )
        word_loss = tf.reduce_sum(word_losses) / tf.to_float(tf.reduce_sum(word_target_length))

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel') or 'projection_w' in v.name]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = 1e-6 * tf.reduce_sum(l2_losses)

        loss = code_loss + word_loss
        loss_with_l2 = loss + l2_loss

    with vs("output"):
        code_outputs_prob = tf.nn.softmax(code_decoder_output)
        code_outputs = tf.argmax(code_outputs_prob, axis=-1)

        word_outputs_prob = tf.nn.softmax(word_decoder_output)
        word_outputs = tf.argmax(word_outputs_prob, axis=-1)

    return dict_to_object({
        'inputs': input_ids,
        'input_len': input_length,
        'copyable_input_ids': copyable_input_ids,
        'code_target': code_target_labels,
        'code_target_len': code_target_length,
        'word_target': word_target_labels,
        'word_target_len': word_target_length,
        'code_outputs': code_outputs,
        'word_outputs': word_outputs,
        'code_outputs_prob': code_outputs_prob,
        'word_outputs_prob': word_outputs_prob,
        'loss': loss,
        'loss_with_l2': loss_with_l2,
        'enable_dropout': use_dropout,
    })
