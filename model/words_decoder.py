import numpy as np
import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.rnn_with_dropout import MultiRnnWithDropout


class WordsDecoder:
    def __init__(self, all_states):
        self.all_states = all_states


class WordsDecoderPlaceholders:
    def __init__(self):
        with variable_scope('placeholders'):
            self.words_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE], 'words_sequence_length')


class WordsDecoderSingleStep:
    def __init__(self, words_logits, words_decoder_new_state):
        self.words_logits = words_logits
        self.words_decoder_new_state = words_decoder_new_state


class WordsDecoderPlaceholdersSingleStep:
    def __init__(self):
        with variable_scope('placeholders'):
            self.words_decoder_inputs = tf.placeholder(tf.float32, [1, WORDS_DECODER_STATE_SIZE])
            self.words_decoder_state = [
                tf.placeholder(tf.float32, [1, WORDS_DECODER_STATE_SIZE])
                for _ in range(WORD_DECODER_LAYERS)
            ]


def build_words_decoder(encoder_last_state):
    with variable_scope("words_decoder") as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        placeholders = WordsDecoderPlaceholders()

        rnn_cell = MultiRnnWithDropout(WORD_DECODER_LAYERS, WORDS_DECODER_STATE_SIZE)

        initial_time = tf.constant(0, dtype=tf.int32)

        maximum_iterations = tf.reduce_max(placeholders.words_sequence_length)

        initial_state = rnn_cell.initial_state(encoder_last_state)

        initial_inputs = rnn_cell.zero_initial_inputs(WORDS_DECODER_STATE_SIZE)

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, WORDS_DECODER_STATE_SIZE]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = rnn_cell(inputs, state)
            outputs_ta = outputs_ta.write(time, cell_output)
            return time + 1, outputs_ta, cell_state, cell_output

        _, final_outputs_ta, final_state, _ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ]
        )
        # time, batch, state_size
        outputs_logits = final_outputs_ta.stack()

        decoder = WordsDecoder(outputs_logits)

        return decoder, placeholders


def build_words_decoder_single_step():
    with variable_scope("words_decoder") as scope:
        rnn_cell = MultiRnnWithDropout(WORD_DECODER_LAYERS, WORDS_DECODER_STATE_SIZE)

        placeholders = WordsDecoderPlaceholdersSingleStep()

        cell_output, cell_state = rnn_cell(placeholders.words_decoder_inputs, placeholders.words_decoder_state)

        def initial_state(encoder_last_state):
            return [encoder_last_state] + [
                np.zeros([1, WORDS_DECODER_STATE_SIZE], np.float32)
                for _ in range(WORD_DECODER_LAYERS - 1)
            ]

        def initial_inputs():
            return np.zeros([1, WORDS_DECODER_STATE_SIZE], np.float32)

        decoder = WordsDecoderSingleStep(cell_output, cell_state)

        initializers = {
            'words_decoder_inputs': initial_inputs,
            'words_decoder_state': initial_state
        }

        return decoder, placeholders, initializers
