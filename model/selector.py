import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.encoder import QueryEncoderPlaceholders, build_query_encoder_for_words
from model.tf_utils import tf_accuracy
from model.words_encoder import build_words_encoder, WordsEncoderPlaceholders


class Selector:
    def __init__(self, scores, score_logits):
        self.scores = scores
        self.score_logits = score_logits


class SelectorPlaceholders:
    def __init__(self, encoder_placeholders, rules_placeholders):
        epc = encoder_placeholders  # type: QueryEncoderPlaceholders
        rpc = rules_placeholders  # type: WordsEncoderPlaceholders

        self.query_ids = epc.query_ids
        self.query_length = epc.query_length

        self.rules_length = rpc.rules_seq_with_pc_len
        self.rules = rpc.rules_seq_with_pc
        self.parent_rules = rpc.parent_rules_seq_with_pc
        self.nodes = rpc.nodes_seq_with_pc

        self.scores_target = tf.placeholder(tf.int32, [])

    def feed(self, query, rules, rule_length, nodes, parent_rules):
        return {
            self.query_ids: [[token] for token in query],
            self.query_length: [len(query)],

            self.rules_length: rule_length,
            self.rules: rules,
            self.nodes: nodes,
            self.parent_rules: parent_rules
        }


def build_selector(query_tokens_count, rules_count, nodes_count):
    with variable_scope('selector'):
        query_encoder, query_encoder_pc = build_query_encoder_for_words(query_tokens_count, batch_size=1)
        rules_encoder, rules_encoder_pc = build_words_encoder(rules_count, nodes_count, batch_size=None)
        placeholders = SelectorPlaceholders(query_encoder_pc, rules_encoder_pc)

        # batch, rules time, rules state size
        _rules_encoder_states = tf.transpose(rules_encoder.all_states, [1, 0, 2])
        # batch, 1, encoder time, rules state size
        prepared_rules_encoder_states = tf.expand_dims(_rules_encoder_states, 1)

        # 1, query time, 1, query state size
        prepared_query_states = tf.expand_dims(query_encoder.all_states, 0)

        assert prepared_rules_encoder_states.shape[-1] == prepared_query_states.shape[-1]

        # batch, query time, rules time, state size
        combined_states = tf.add(prepared_rules_encoder_states, prepared_query_states, name='combine_states')
        bounded_combined_states = tf.nn.relu(combined_states)
        bounded_combined_states = tf.nn.dropout(bounded_combined_states, 0.8)

        with variable_scope('variables'):
            rules_weight = tf.get_variable(
                name='kernel_rules',
                shape=[WORDS_DECODER_STATE_SIZE],
                dtype=tf.float32
            )
            rules_bias = tf.get_variable(
                name='bias_rules',
                shape=[WORDS_DECODER_STATE_SIZE],
                dtype=tf.float32
            )

            weight = tf.get_variable(
                name='kernel',
                shape=[128],
                dtype=tf.float32
            )

            one_more_weight = tf.get_variable(
                name='kernel_1',
                shape=[WORDS_DECODER_STATE_SIZE, 128],
                dtype=tf.float32
            )
            one_more_bias = tf.get_variable(
                name='bias_1',
                shape=[128],
                dtype=tf.float32
            )

        # batch, rules_time, state size
        weighed_rules_states = tf.einsum('aibk,k->abk', bounded_combined_states, rules_weight)
        biased_rules_states = tf.add(weighed_rules_states, rules_bias)
        rules_states = tf.nn.relu(biased_rules_states)
        rules_states = tf.nn.dropout(rules_states, 0.8)

        one_more_layer_x = tf.einsum('abi,ij->abj', rules_states, one_more_weight)
        one_more_layer_y = tf.add(one_more_layer_x, one_more_bias)
        one_more_layer = tf.nn.relu(one_more_layer_y)
        one_more_layer = tf.nn.dropout(one_more_layer, 0.8)

        # batch, rules_time, encoder_time
        scores_logits = tf.einsum('abi,i->a', one_more_layer, weight, name='scores_logits')

        with variable_scope('scores_softmax'):
            scores = tf.nn.softmax(scores_logits)

    selector = Selector(scores, scores_logits)
    return selector, placeholders


def build_selector_loss(selector, placeholders):
    # type: (Selector, SelectorPlaceholders) -> [tf.Tensor, tf.Tensor]

    raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=placeholders.scores_target,
        logits=selector.score_logits
    )
    loss = tf.reduce_sum(raw_loss)

    predictions = tf.argmax(selector.scores)
    accuracy = tf_accuracy(predictions, placeholders.scores_target)

    with variable_scope('l2_loss'):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel')]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = L2_COEFFICIENT * tf.reduce_sum(l2_losses)
        loss_with_l2 = loss + l2_loss

    return loss_with_l2, accuracy
