import numpy as np
import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.tf_utils import *
from model.rnn_with_dropout import MultiRnnWithDropout


class RulesDecoderPlaceholders:
    def __init__(self):
        with variable_scope('placeholders'):
            self.rules_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'rules_target')
            self.rules_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE], 'rules_target_length')


class RulesDecoder(object):
    def __init__(self, rules, rules_logits):
        self.rules = rules
        self.rules_logits = rules_logits


class RulesDecoderSingleStep(RulesDecoder):
    def __init__(self, rules, rules_logits, rules_decoder_new_state):
        super(RulesDecoderSingleStep, self).__init__(rules, rules_logits)
        self.rules_decoder_new_state = rules_decoder_new_state


class RulesDecoderPlaceholdersSingleStep:
    def __init__(self, rules_count):
        with variable_scope('placeholders'):
            self.inputs = tf.placeholder(tf.float32, [None, rules_count])
            self.states = tuple([
                tf.placeholder(tf.float32, [None, RULES_DECODER_STATE_SIZE])
                for _ in range(RULES_DECODER_LAYERS)
            ])
            self.query_encoder_all_states = tf.placeholder(tf.float32, [None, None, RULES_DECODER_STATE_SIZE])


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
        combined_shape = RULES_QUERY_ENCODER_STATE_SIZE * 2 + RULES_DECODER_STATE_SIZE
        W_c = tf.get_variable('W_c', shape=[combined_shape, RULES_DECODER_ATTENTION_SIZE], dtype=tf.float32)
        multiplied = tf.matmul(combined, W_c, name='attn_matmul')
        attention_vec = tf.tanh(multiplied)
        return attention_vec


def build_rules_decoder(query_encoder, rules_count):
    with variable_scope("rules_decoder") as scope:
        placeholders = RulesDecoderPlaceholders()

        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        maximum_iterations = tf.reduce_max(placeholders.rules_sequence_length)

        rnn_cell = MultiRnnWithDropout(RULES_DECODER_LAYERS, RULES_DECODER_STATE_SIZE)
        outputs_projection = tf.layers.Dense(rules_count, name='rules_output_projection')

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_state = rnn_cell.initial_state(query_encoder.last_state)
        initial_inputs = rnn_cell.zero_initial_inputs(rules_count)

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, rules_count]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = rnn_cell(inputs, state)
            state_with_attention = attention(cell_output, query_encoder.all_states)
            projected_outputs = outputs_projection(state_with_attention)
            outputs_ta = outputs_ta.write(time, projected_outputs)
            probability_outputs = tf.nn.softmax(projected_outputs)
            return time + 1, outputs_ta, cell_state, probability_outputs

        _, final_outputs_ta, final_state, _ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ]
        )

        # time, batch, rules_count
        outputs_logits = final_outputs_ta.stack()
        outputs = tf.nn.softmax(outputs_logits)

        decoder = RulesDecoder(outputs, outputs_logits)

        return decoder, placeholders


def build_rules_decoder_single_step(rules_count):
    with variable_scope("rules_decoder") as scope:
        placeholders = RulesDecoderPlaceholdersSingleStep(rules_count)

        rnn_cell = MultiRnnWithDropout(RULES_DECODER_LAYERS, RULES_DECODER_STATE_SIZE)
        outputs_projection = tf.layers.Dense(rules_count, name='rules_output_projection')

        cell_output, cell_state = rnn_cell(placeholders.inputs, placeholders.states)
        state_with_attention = attention(cell_output, placeholders.query_encoder_all_states)
        projected_outputs = outputs_projection(state_with_attention)

        # time, batch, rules_count
        outputs = tf.nn.softmax(projected_outputs)

        def initial_state(encoder_last_state):
            return [encoder_last_state] + [
                np.zeros([1, RULES_DECODER_STATE_SIZE], np.float32)
                for _ in range(RULES_DECODER_LAYERS - 1)
            ]

        def initial_inputs():
            return np.zeros([1, rules_count], np.float32)

        decoder = RulesDecoderSingleStep(outputs, projected_outputs, cell_state)

        initializers = {
            'rules_decoder_inputs': initial_inputs,
            'rules_decoder_state': initial_state
        }

        return decoder, placeholders, initializers


def build_rules_loss(decoder, decoder_placeholders):
    # type: (RulesDecoder, RulesDecoderPlaceholders) -> [dict, dict]
    with variable_scope("rules_decoder_loss"):
        raw_rules_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_placeholders.rules_target,
            logits=decoder.rules_logits
        )
        loss_mask = tf_length_mask(decoder_placeholders.rules_sequence_length)
        rules_loss = tf_mask_gracefully(raw_rules_loss, loss_mask, sum_result=True)

    with variable_scope('stats'):
        scaled_logits = tf.nn.softmax(decoder.rules_logits)
        results = tf.argmax(scaled_logits, axis=-1)
        rules_accuracy = tf_accuracy(
            predicted=results,
            target=decoder_placeholders.rules_target,
            mask=loss_mask
        )

    stats = {
        'rules_accuracy': rules_accuracy
    }

    loss = {
        'rules_loss': rules_loss,
    }
    return loss, stats
