import tensorflow as tf

from tensorflow import variable_scope as vs

import utilss
from copy_rnn_cell import CopyMultiRnnCell
from rnn_cell_builder import get_rnn_cell
from utilss import dict_to_object

from net_conf import *


# from new_res.model_1_conf import *

# from results.model_9_net_conf import *


def dynamic_decode(
        decoder_cell,
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

        initial_state = decoder_cell.get_initial_state()
        initial_inputs = decoder_cell.get_initial_input()

        initial_outputs_ta = decoder_cell.initialize_outputs_ta(maximum_iterations)

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = decoder_cell(inputs, state)
            outputs_ta = decoder_cell.write_outputs(time, outputs_ta, cell_output)
            return time + 1, outputs_ta, cell_state, cell_output

        _, final_outputs_ta, final_state, *_ = tf.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs = decoder_cell.finalize_outputs(final_outputs_ta)

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

        code_target_labels = tf.placeholder(tf.int32, [None, batch_size], 'code_targets')
        code_target_length = tf.placeholder(tf.int32, [batch_size], 'output_length')
        word_target_labels = tf.placeholder(tf.int32, [None, batch_size], 'word_targets')
        word_target_length = tf.placeholder(tf.int32, [batch_size], 'word_length')

        copyable_input_ids = tf.placeholder(tf.int32, [None, batch_size], 'copyable_input_ids')
        copy_targets = tf.placeholder(tf.int32, [None, batch_size], 'copy_targets')

        unknown_word_label = tf.placeholder(tf.int32, [], 'unknown_word')

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
            output_size=code_output_num_tokens,
            batch_size=batch_size,
            dropout_prob=dropout_prob,
            output_projection=True,
            projection_size=code_output_num_tokens,
            attention=CODE_DECODER_ATTENTION,
            attention_source=attention_source_states,
            attention_vec_size=decoder_state_size,
        )
        decoder_cell.set_initial_state(encoder_result)
        code_decoder_output, code_decoder_state = dynamic_decode(
            decoder_cell=decoder_cell,
            maximum_iterations=tf.reduce_max(code_target_length),
            parallel_iterations=1,
            scope=decoder_scope
        )

    with vs("word_decoder") as decoder_scope:
        decoder_state_size = ENCODER_STATE_SIZE * 2

        decoder_cell = CopyMultiRnnCell(
            num_layers=WORD_DECODER_LAYERS,
            state_size=decoder_state_size,
            output_size=word_output_num_tokens,
            batch_size=batch_size,
            dropout_prob=dropout_prob,
            encoder_states=encoder_output,
            encoder_seq_length=tf.reduce_max(input_length),
            encoder_state_size=decoder_state_size,
            encoder_input_ids=copyable_input_ids,
        )
        decoder_cell.set_initial_state(encoder_result)

        word_decoder_output, word_decoder_state = dynamic_decode(
            decoder_cell=decoder_cell,
            maximum_iterations=tf.reduce_max(word_target_length),
            parallel_iterations=1,
            scope=decoder_scope
        )
        word_decoder_generate, word_decoder_copy = word_decoder_output

    with vs("loss"):
        code_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=code_decoder_output,
            labels=code_target_labels,
        )

        # Calculate the average log perplexity
        code_loss = tf.reduce_sum(code_losses) / tf.to_float(tf.reduce_sum(code_target_length))

        word_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=word_decoder_generate,
            labels=word_target_labels,
        )

        copy_outputs_mask = tf.not_equal(copy_targets, -1)
        copy_is_empty = tf.equal(tf.count_nonzero(copy_outputs_mask), 0)

        copy_outputs_for_losses = tf.boolean_mask(word_decoder_copy, copy_outputs_mask)
        copy_outputs_labels = tf.boolean_mask(copy_targets, copy_outputs_mask)

        copy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=copy_outputs_for_losses,
            labels=copy_outputs_labels
        )

        copy_loss = tf.reduce_mean(copy_losses)
        copy_loss = tf.cond(copy_is_empty, lambda: 0.0, lambda: copy_loss)

        unknown_word_labels = tf.fill(tf.shape(word_target_labels), unknown_word_label)
        unknown_word_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=word_decoder_generate,
            labels=unknown_word_labels,
        )

        unknown_word_losses = tf.where(copy_outputs_mask, unknown_word_losses, tf.zeros_like(unknown_word_losses))
        word_loss_with_unknown = (word_losses + unknown_word_losses) / 2
        word_losses = tf.where(copy_outputs_mask, word_loss_with_unknown, word_losses)

        word_loss = tf.reduce_sum(word_losses) / tf.to_float(tf.reduce_sum(word_target_length))

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel') or 'projection_w' in v.name]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = 1e-6 * tf.reduce_sum(l2_losses)

        loss = code_loss + word_loss + copy_loss
        loss_with_l2 = loss + l2_loss

        loss_with_l2 = tf.Print(loss_with_l2, [loss_with_l2], 'l2: ')

    with vs("output"):
        code_outputs_prob = tf.nn.softmax(code_decoder_output)
        code_outputs = tf.argmax(code_outputs_prob, axis=-1)

        word_outputs_prob = tf.nn.softmax(word_decoder_generate)
        word_outputs = tf.argmax(word_outputs_prob, axis=-1)

        copy_outputs = tf.argmax(word_decoder_copy, axis=-1)

    return dict_to_object({
        'inputs': input_ids,
        'input_len': input_length,
        'copyable_input_ids': copyable_input_ids,
        'code_target': code_target_labels,
        'code_target_len': code_target_length,
        'word_target': word_target_labels,
        'word_target_len': word_target_length,
        'unknown_word': unknown_word_label,
        'copy_target': copy_targets,
        'code_outputs': code_outputs,
        'word_outputs': word_outputs,
        'copy_outputs': copy_outputs,
        'code_outputs_prob': code_outputs_prob,
        'word_outputs_prob': word_outputs_prob,
        'loss': loss,
        'loss_with_l2': loss_with_l2,
        'enable_dropout': use_dropout,
    })
