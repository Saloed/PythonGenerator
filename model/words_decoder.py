import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.rnn_with_dropout import MultiRnnWithDropout
from utils import dict_to_object


def build_words_decoder(encoder_last_state):
    with variable_scope("words_decoder") as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        with variable_scope('placeholders'):
            words_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE], 'words_sequence_length')

        rnn_cell = MultiRnnWithDropout(WORD_DECODER_LAYERS, DECODER_STATE_SIZE)

        initial_time = tf.constant(0, dtype=tf.int32)

        maximum_iterations = tf.reduce_max(words_sequence_length)

        initial_state = rnn_cell.initial_state(encoder_last_state)

        initial_inputs = rnn_cell.zero_initial_inputs(DECODER_STATE_SIZE)

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, DECODER_STATE_SIZE]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = rnn_cell(inputs, state)
            outputs_ta = outputs_ta.write(time, cell_output)
            return time + 1, outputs_ta, cell_state, cell_output

        _, final_outputs_ta, final_state, *_ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ]
        )
        # time, batch, state_size
        outputs_logits = final_outputs_ta.stack()

        placeholders = {
            'words_sequence_length': words_sequence_length
        }

        decoder = dict_to_object({
            'all_states': outputs_logits
        }, placeholders)

        return decoder, placeholders
