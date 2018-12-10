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
            self.nodes = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'nodes')
            self.parent_rules = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'parent_rules')
            self.parent_rules_t = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'parent_rules_t')


class RulesDecoder(object):
    def __init__(self, rules, rules_logits):
        self.rules = rules
        self.rules_logits = rules_logits


class RulesDecoderSingleStep(RulesDecoder):
    def __init__(self, rules, rules_logits, rules_decoder_new_state, attention_context):
        super(RulesDecoderSingleStep, self).__init__(rules, rules_logits)
        self.attention_context = attention_context
        self.rules_decoder_new_state = rules_decoder_new_state

    def fetch_all(self):
        return [
            self.rules_decoder_new_state,
            self.attention_context,
            self.rules_logits,
            self.rules
        ]


class RulesDecoderPlaceholdersSingleStep:
    def __init__(self):
        with variable_scope('placeholders'):
            self.states = tuple([
                tf.placeholder(tf.float32, [None, RULES_DECODER_STATE_SIZE])
                for _ in range(RULES_DECODER_LAYERS)
            ])
            self.query_encoder_all_states = tf.placeholder(tf.float32, [None, None, RULES_DECODER_STATE_SIZE])
            self.query_length = tf.placeholder(tf.int32, [None])

            self.previous_rule = tf.placeholder(tf.int32, [None])
            self.frontier_node = tf.placeholder(tf.int32, [None])
            self.parent_rule = tf.placeholder(tf.int32, [None])
            self.parent_rule_state = tf.placeholder(tf.float32, [None, RULES_DECODER_STATE_SIZE])

            self.attention_context = tf.placeholder(tf.float32, [None, RULES_DECODER_ATTENTION_SIZE])

    def feed(self, states, attention_ctx, query_states, query_length, prev_rule, frontier_node, parent_rule,
             parent_rule_state):
        return {
            self.states: states,
            self.query_encoder_all_states: query_states,
            self.query_length: query_length,
            self.attention_context: attention_ctx,
            self.previous_rule: prev_rule,
            self.frontier_node: frontier_node,
            self.parent_rule: parent_rule,
            self.parent_rule_state: parent_rule_state
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
        combined_shape = RULES_QUERY_ENCODER_STATE_SIZE * 2 + RULES_DECODER_STATE_SIZE
        W_c = tf.get_variable('W_c', shape=[combined_shape, RULES_DECODER_ATTENTION_SIZE], dtype=tf.float32)
        multiplied = tf.matmul(combined, W_c, name='attn_matmul')
        attention_vec = tf.tanh(multiplied)
        return attention_vec


class BahdanauAttention:
    def __init__(self):
        with variable_scope("attention"):
            self.query_layer = tf.layers.Dense(RULES_DECODER_ATTENTION_SIZE, name="query_layer", use_bias=False)
            self.memory_layer = tf.layers.Dense(RULES_DECODER_ATTENTION_SIZE, name="memory_layer", use_bias=False)
            self.v = tf.get_variable("attention_v", [RULES_DECODER_ATTENTION_SIZE])

    def __call__(self, source_states, source_state_length, hidden_states):
        with variable_scope("attention"):
            memory = tf.transpose(source_states, [1, 0, 2])
            mask = tf.sequence_mask(source_state_length, dtype=tf.float32)
            mask = tf.expand_dims(mask, 2)
            values = memory * mask
            keys = self.memory_layer(values)

            processed_query = self.query_layer(hidden_states)
            processed_query = tf.expand_dims(processed_query, 1)
            score = tf.reduce_sum(self.v * tf.tanh(keys + processed_query), [2])
            alignments = tf.nn.softmax(score)
            weight = tf.expand_dims(alignments, 2)
            weighted_keys = keys * weight
            attention_vec = tf.reduce_sum(weighted_keys, axis=1)

        return attention_vec


class _RulesDecoderCell:
    def __init__(self, rules_count, nodes_count):
        self.rules_count = rules_count
        self.nodes_count = nodes_count
        self.rnn_cell = MultiRnnWithDropout(RULES_DECODER_LAYERS, RULES_DECODER_STATE_SIZE)

        self.outputs_projection = tf.layers.Dense(rules_count, name='rules_output_projection')

        self.node_embedding = tf.get_variable(
            name='node_embedding',
            shape=[nodes_count, RULES_DECODER_EMBEDDING_SIZE],
            dtype=tf.float32
        )

        self.rule_embedding = tf.get_variable(
            name='rule_embedding',
            shape=[rules_count, RULES_DECODER_EMBEDDING_SIZE],
            dtype=tf.float32
        )

        self.attention = BahdanauAttention()

    def __call__(self, previous_rule_id, parent_rule_ids, frontier_node_ids, parent_states,
                 context, state, query_encoder_all_states, query_length):
        rule_condition = tf.not_equal(previous_rule_id, -1)
        parent_rule_condition = tf.not_equal(parent_rule_ids, -1)
        frontier_node_condition = tf.not_equal(frontier_node_ids, -1)

        frontier_nodes = tf_conditional_lookup(frontier_node_condition, self.node_embedding, frontier_node_ids)
        prev_rules = tf_conditional_lookup(rule_condition, self.rule_embedding, previous_rule_id)
        parent_rules = tf_conditional_lookup(parent_rule_condition, self.rule_embedding, parent_rule_ids)

        inputs = tf.concat([prev_rules, context, parent_rules, parent_states, frontier_nodes], axis=-1)

        cell_output, cell_state = self.rnn_cell(inputs, state)

        # attention_context = attention(cell_output, query_encoder_all_states)
        attention_context = self.attention(query_encoder_all_states, query_length, cell_output)

        projected_outputs = self.outputs_projection(cell_output)

        probability_outputs = tf.nn.softmax(projected_outputs)
        rule = tf.argmax(probability_outputs, axis=-1, output_type=tf.int32)

        return cell_output, cell_state, attention_context, projected_outputs, probability_outputs, rule


def build_rules_decoder(query_encoder, query_encoder_pc, rules_count, nodes_count):
    with variable_scope("rules_decoder") as scope:
        placeholders = RulesDecoderPlaceholders()

        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        decoder_cell = _RulesDecoderCell(rules_count, nodes_count)

        maximum_iterations = tf.reduce_max(placeholders.rules_sequence_length)

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_rules = -tf.ones([BATCH_SIZE], dtype=tf.int32)
        initial_context = tf.zeros([BATCH_SIZE, RULES_DECODER_ATTENTION_SIZE])

        initial_state = decoder_cell.rnn_cell.initial_state(query_encoder.last_state)

        initial_last_state = initial_state[-1]

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, rules_count]
        )

        initial_states_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, RULES_DECODER_STATE_SIZE],
            clear_after_read=False,
        )

        def condition(_time, *args):
            return _time < maximum_iterations

        def body(time, outputs_ta, states_ta, state, context, previous_rule_id):
            frontier_node_ids = placeholders.nodes[time]
            parent_rule_ids = placeholders.parent_rules[time]
            parent_rule_time = placeholders.parent_rules_t[time]

            parent_rule_condition = tf.not_equal(parent_rule_ids, -1)
            _parent_states = tf_conditional_ta_lookup(states_ta, parent_rule_condition, parent_rule_time,
                                                      initial_last_state)

            _unbatched_states = []
            for batch_id in range(BATCH_SIZE):
                _unbatched_states.append(_parent_states[batch_id][batch_id])
            parent_states = tf.stack(_unbatched_states)

            cell_output, cell_state, attention_context, projected_outputs, probability_outputs, rule = decoder_cell(
                previous_rule_id=previous_rule_id,
                parent_rule_ids=parent_rule_ids,
                frontier_node_ids=frontier_node_ids,
                parent_states=parent_states,
                context=context,
                state=state,
                query_encoder_all_states=query_encoder.all_states,
                query_length=query_encoder_pc.query_length
            )
            states_ta = states_ta.write(time, cell_output)
            outputs_ta = outputs_ta.write(time, projected_outputs)

            return time + 1, outputs_ta, states_ta, cell_state, attention_context, rule

        _, final_outputs_ta, _, final_state, _, _ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_states_ta, initial_state, initial_context, initial_rules
            ]
        )

        # time, batch, rules_count
        outputs_logits = final_outputs_ta.stack()
        outputs = tf.nn.softmax(outputs_logits)

        decoder = RulesDecoder(outputs, outputs_logits)

        return decoder, placeholders


def build_rules_decoder_single_step(rules_count, nodes_count):
    with variable_scope("rules_decoder") as scope:
        placeholders = RulesDecoderPlaceholdersSingleStep()

        decoder_cell = _RulesDecoderCell(rules_count, nodes_count)
        cell_output, cell_state, attention_context, projected_outputs, probability_outputs, rule = decoder_cell(
            previous_rule_id=placeholders.previous_rule,
            parent_rule_ids=placeholders.parent_rule,
            frontier_node_ids=placeholders.frontier_node,
            parent_states=placeholders.parent_rule_state,
            context=placeholders.attention_context,
            state=placeholders.states,
            query_encoder_all_states=placeholders.query_encoder_all_states,
            query_length=placeholders.query_length
        )

        decoder = RulesDecoderSingleStep(probability_outputs, projected_outputs, cell_state, attention_context)

        def initial_state(encoder_last_state):
            return [encoder_last_state] + [
                np.zeros([1, RULES_DECODER_STATE_SIZE], np.float32)
                for _ in range(RULES_DECODER_LAYERS - 1)
            ]

        def initial_context():
            return np.zeros([1, RULES_DECODER_ATTENTION_SIZE], np.float32)

        initializers = {
            'rules_decoder_state': initial_state,
            'rules_decoder_context': initial_context
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
