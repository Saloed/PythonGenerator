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
    def __init__(self, words_logits, words_decoder_new_state, attention_context):
        self.words_logits = words_logits
        self.new_state = words_decoder_new_state
        self.new_attention = attention_context

    def decoder_fetches(self):
        return [
            self.words_logits,
            self.new_state,
            self.new_attention
        ]


class WordsDecoderPlaceholdersSingleStep:
    def __init__(self):
        with variable_scope('placeholders'):
            self.states = tuple([
                tf.placeholder(tf.float32, [None, WORDS_DECODER_STATE_SIZE])
                for _ in range(WORDS_DECODER_LAYERS)
            ])
            self.encoder_all_states = tf.placeholder(tf.float32, [None, None, WORDS_DECODER_STATE_SIZE])
            self.attention_context = tf.placeholder(tf.float32, [None, WORDS_DECODER_ATTENTION_SIZE])

    def decoder_feed(self, states, context, encoder_states):
        return {
            self.states: states,
            self.encoder_all_states: encoder_states,
            self.attention_context: context
        }


# Luong score
def compute_alphas(h_t, h_s):
    _score = tf.matmul(h_t, h_s, transpose_b=True, name='score_matmul')
    score = tf.squeeze(_score, [1])
    alpha_ts = tf.nn.softmax(score)
    alpha_ts = tf.expand_dims(alpha_ts, 2)
    return alpha_ts


def attention(
        hidden_state,
        source_states
):
    with variable_scope("attention"):
        h_s = tf.transpose(source_states, [1, 0, 2])
        h_t = tf.expand_dims(hidden_state, axis=1)
        alpha_ts = compute_alphas(h_t, h_s)
        weighted_sources = alpha_ts * h_s
        context = tf.reduce_sum(weighted_sources, axis=1)
        combined = tf.concat([context, hidden_state], axis=1)
        combined_shape = RULES_ENCODER_STATE_SIZE * 2 + WORDS_DECODER_STATE_SIZE
        W_c = tf.get_variable('W_c', shape=[combined_shape, WORDS_DECODER_ATTENTION_SIZE], dtype=tf.float32)
        multiplied = tf.matmul(combined, W_c, name='attn_matmul')
        attention_vec = tf.tanh(multiplied)
        return attention_vec


class _WordsDecoderCell:
    def __init__(self):
        self.rnn_cell = MultiRnnWithDropout(WORDS_DECODER_LAYERS, WORDS_DECODER_STATE_SIZE)

    def __call__(self, attention_context, state, encoder_all_states):
        cell_output, cell_state = self.rnn_cell(attention_context, state)
        attention_context = attention(cell_output, encoder_all_states)
        return cell_output, cell_state, attention_context


def build_words_decoder(words_encoder):
    with variable_scope("words_decoder") as scope:
        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        decoder_cell = _WordsDecoderCell()

        placeholders = WordsDecoderPlaceholders()

        initial_time = tf.constant(0, dtype=tf.int32)

        maximum_iterations = tf.reduce_max(placeholders.words_sequence_length)

        initial_state = decoder_cell.rnn_cell.initial_state(words_encoder.last_state)
        initial_context = tf.zeros([BATCH_SIZE, WORDS_DECODER_ATTENTION_SIZE])

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, WORDS_DECODER_STATE_SIZE]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, attention_context):
            cell_output, cell_state, attn_ctx = decoder_cell(attention_context, state, words_encoder.all_states)
            outputs_ta = outputs_ta.write(time, cell_output)
            return time + 1, outputs_ta, cell_state, attn_ctx

        _, final_outputs_ta, final_state, _ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_context
            ]
        )
        # time, batch, state_size
        outputs_logits = final_outputs_ta.stack()

        decoder = WordsDecoder(outputs_logits)

        return decoder, placeholders


def build_words_decoder_single_step():
    with variable_scope("words_decoder") as scope:
        placeholders = WordsDecoderPlaceholdersSingleStep()
        decoder_cell = _WordsDecoderCell()
        cell_output, cell_state, attn_ctx = decoder_cell(placeholders.attention_context, placeholders.states,
                                                         placeholders.encoder_all_states)

        decoder = WordsDecoderSingleStep(cell_output, cell_state, attn_ctx)

        def initial_state(encoder_last_state):
            return [encoder_last_state] + [
                np.zeros([1, WORDS_DECODER_STATE_SIZE], np.float32)
                for _ in range(WORDS_DECODER_LAYERS - 1)
            ]

        def initial_context():
            return np.zeros([1, WORDS_DECODER_ATTENTION_SIZE], np.float32)

        initializers = {
            'words_decoder_context': initial_context,
            'words_decoder_state': initial_state
        }

        return decoder, placeholders, initializers
