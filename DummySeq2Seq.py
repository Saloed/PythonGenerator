import tensorflow as tf

from tensorflow import variable_scope as vs


def dynamic_decode(
        decoder_cell,
        batch_size,
        encoder_results,
        decoder_initial_input,
        output_size,
        maximum_iterations=None,
        parallel_iterations=32,
        swap_memory=False,
        scope=None
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

        initial_state = list(decoder_cell.zero_state(batch_size, tf.float32))
        initial_state[0] = encoder_results
        initial_state = tuple(initial_state)

        initial_inputs = decoder_initial_input

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[batch_size, output_size]
        )

        output_projection_w = tf.get_variable(
            name='output_projection_w',
            shape=[decoder_cell.output_size, output_size],
            dtype=tf.float32,
        )

        output_projection_b = tf.get_variable(
            name='output_projection_b',
            shape=[output_size],
            dtype=tf.float32,
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = decoder_cell(inputs, state)
            projected_output = tf.nn.relu_layer(cell_output, output_projection_w, output_projection_b)
            outputs_ta = outputs_ta.write(time, projected_output)
            return time + 1, outputs_ta, cell_state, projected_output

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


def build_model(batch_size, input_num_tokens, output_num_tokens, is_in_train_mode):

    with vs("inputs"):
        input_ids = tf.placeholder(tf.int32, [None, batch_size])
        input_weights = tf.placeholder(tf.float32, [None, batch_size])
        input_length = tf.placeholder(tf.int32, [batch_size])
        target_labels = tf.placeholder(tf.int32, [None, batch_size])
        target_length = tf.placeholder(tf.int32, [batch_size])

        encoder_input_size = 128

        embedding = tf.get_variable(
            name='input_embedding',
            shape=[input_num_tokens, encoder_input_size],
            dtype=tf.float32
        )

        prepared_inputs = tf.nn.embedding_lookup(embedding, input_ids)
        weighted_inputs = input_weights * prepared_inputs

    with vs("encoder") as encoder_scope:
        encoder_state_size = 128
        encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(encoder_state_size) for _ in range(2)]
        encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(encoder_state_size) for _ in range(2)]

        from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
        encoder_output, encoder_state_fw, encoder_state_bw = stack_bidirectional_dynamic_rnn(
            cells_fw=encoder_fw_internal_cells,
            cells_bw=encoder_bw_internal_cells,
            inputs=weighted_inputs,
            sequence_length=input_length,
            time_major=True,
            dtype=tf.float32,
            scope=encoder_scope,
        )
        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        # encoder_fw_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_fw_internal_cells, state_is_tuple=False)
        # encoder_bw_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_bw_internal_cells, state_is_tuple=False)
        #
        # encoder_output, (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        #     cell_fw=encoder_fw_cell,
        #     cell_bw=encoder_bw_cell,
        #     inputs=inputs,
        #     sequence_length=input_length,
        #     time_major=True,
        #     dtype=tf.float32,
        #     scope=encoder_scope,
        # )
        # final_encoder_state_fw = encoder_state_fw[:, -encoder_state_size:]
        # final_encoder_state_bw = encoder_state_bw[:, -encoder_state_size:]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

    with vs("decoder") as decoder_scope:
        decoder_state_size = encoder_state_size * 2
        decoder_internal_cells = [tf.nn.rnn_cell.GRUCell(decoder_state_size) for _ in range(2)]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_internal_cells)
        decoder_initial_input = [tf.one_hot(output_num_tokens - 2, output_num_tokens) for _ in range(batch_size)]
        decoder_initial_input = tf.stack(decoder_initial_input, axis=0)
        decoder_output, decoder_state = dynamic_decode(
            decoder_cell=decoder_cell,
            batch_size=batch_size,
            decoder_initial_input=decoder_initial_input,
            encoder_results=encoder_result,
            output_size=output_num_tokens,
            maximum_iterations=tf.reduce_max(target_length),
            parallel_iterations=1,
            scope=decoder_scope,
        )

    with vs("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=decoder_output,
            labels=target_labels,
        )

        # Calculate the average log perplexity
        loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(target_length))

    with vs("output"):
        outputs = tf.nn.softmax(decoder_output)
        outputs = tf.argmax(outputs, axis=-1)

    return input_ids, input_length, target_labels, target_length, outputs, loss
