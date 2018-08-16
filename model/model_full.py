import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.copy_mechanism import build_copy_mechanism, build_words_loss, CopyMechanismPlaceholders
from model.encoder import build_query_encoder_for_rules, build_query_encoder_for_words, QueryEncoderPlaceholders
from model.rules_decoder import build_rules_decoder, build_rules_loss, RulesDecoderPlaceholders
from model.words_decoder import build_words_decoder, WordsDecoderPlaceholders
from model.words_encoder import build_words_encoder, WordsEncoderPlaceholders
from utils import dict_to_object, Magic


class RulesModelPlaceholders(Magic, QueryEncoderPlaceholders, RulesDecoderPlaceholders):
    pass


class WordsModelPlaceholders(Magic, QueryEncoderPlaceholders, WordsEncoderPlaceholders, WordsDecoderPlaceholders,
                             CopyMechanismPlaceholders):
    pass


class FullModel(object):
    def __init__(self, loss, loss_with_l2, stats, updates):
        self.loss = loss
        self.loss_with_l2 = loss_with_l2
        self.stats = stats
        self.updates = updates


class RulesModel(FullModel):
    def __init__(self, placeholders, loss, loss_with_l2, stats, updates=None):
        super(RulesModel, self).__init__(loss, loss_with_l2, stats, updates)
        self.placeholders = placeholders  # type: RulesModelPlaceholders


class WordsModel(FullModel):
    def __init__(self, placeholders, loss, loss_with_l2, stats, updates=None):
        super(WordsModel, self).__init__(loss, loss_with_l2, stats, updates)
        self.placeholders = placeholders  # type: WordsModelPlaceholders


def build_rules_model(rules_count, query_tokens_count, nodes_count):
    query_encoder, query_encoder_pc = build_query_encoder_for_rules(query_tokens_count, batch_size=BATCH_SIZE)
    rules_decoder, rules_decoder_pc = build_rules_decoder(query_encoder, rules_count, nodes_count)

    with variable_scope('loss'):
        rules_loss, rules_stats = build_rules_loss(rules_decoder, rules_decoder_pc)

        stacked_rules_loss = tf.stack(list(rules_loss.values()))

        loss = tf.reduce_sum(stacked_rules_loss)

    with variable_scope('l2_loss'):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel')]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = L2_COEFFICIENT * tf.reduce_sum(l2_losses)
        loss_with_l2 = loss + l2_loss

    placeholders = RulesModelPlaceholders(query_encoder_pc, rules_decoder_pc)

    stats = dict_to_object(
        rules_stats,
    )
    model = RulesModel(placeholders, loss, loss_with_l2, stats)
    return model


def build_words_model(query_tokens_count, rules_count, nodes_count, words_count):
    words_query_encoder, words_query_encoder_pc = build_query_encoder_for_words(query_tokens_count,
                                                                                batch_size=BATCH_SIZE)
    words_encoder, words_encoder_pc = build_words_encoder(rules_count, nodes_count)
    words_decoder, words_decoder_pc = build_words_decoder(words_encoder)
    copy_mechanism, copy_mechanism_pc = build_copy_mechanism(words_query_encoder.all_states, words_decoder.all_states,
                                                             words_count)

    with variable_scope('loss'):
        words_loss, words_stats = build_words_loss(copy_mechanism, words_decoder_pc, copy_mechanism_pc)

        stacked_words_loss = tf.stack(list(words_loss.values()))

        loss = tf.reduce_sum(stacked_words_loss)

    with variable_scope('l2_loss'):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel')]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = L2_COEFFICIENT * tf.reduce_sum(l2_losses)
        loss_with_l2 = loss + l2_loss

    placeholders = WordsModelPlaceholders(words_query_encoder_pc, words_encoder_pc, words_decoder_pc, copy_mechanism_pc)

    stats = dict_to_object(
        words_stats
    )
    model = WordsModel(placeholders, loss, loss_with_l2, stats)
    return model


def apply_optimizer(model_instance, is_rules_model):
    if is_rules_model:
        optimizer = tf.train.AdamOptimizer(0.0005)
        updates = optimizer.minimize(model_instance.loss)
    else:
        optimizer = tf.train.RMSPropOptimizer(0.001)
        # optimizer = tf.train.AdamOptimizer(0.0005)
        updates = optimizer.minimize(model_instance.loss_with_l2)
    model_instance.updates = updates


def make_rules_feed(data, model_instance):
    # type: (tuple, RulesModel) -> dict
    (q_ids, q_length), (rules, rules_length), nodes, parent_rules, parent_rules_t = data
    pc = model_instance.placeholders
    return {
        pc.query_ids: q_ids,
        pc.query_length: q_length,
        pc.rules_sequence_length: rules_length,
        pc.rules_target: rules,
        pc.nodes: nodes,
        pc.parent_rules: parent_rules,
        pc.parent_rules_t: parent_rules_t
    }


def make_words_feed(data, model_instance):
    # type: (tuple, WordsModel) -> dict
    (q_ids, q_length), (rules, rules_length), nodes, parent_rules, parent_rules_t, words, words_mask, copy, copy_mask, (
        gen_or_copy, words_length) = data
    pc = model_instance.placeholders
    return {
        pc.query_ids: q_ids,
        pc.query_length: q_length,

        pc.generate_token_target: words,
        pc.generate_token_target_mask: words_mask,
        pc.copy_target: copy,
        pc.copy_target_mask: copy_mask,
        pc.generate_or_copy_target: gen_or_copy,
        pc.words_sequence_length: words_length,

        pc.rules_seq_with_pc: rules,
        pc.rules_seq_with_pc_len: rules_length,
        pc.parent_rules_seq_with_pc: parent_rules,
        pc.nodes_seq_with_pc: nodes
    }


def make_fetches(model_instance, is_train):
    # type: (FullModel, bool) -> list
    fetches = [model_instance.loss_with_l2]
    if is_train:
        fetches += [model_instance.updates]
    return fetches


def make_rules_stats_fetches(model_instance):
    # type: (RulesModel) -> list
    st = model_instance.stats
    return [
        st.rules_accuracy,
    ]


def make_words_stats_fetches(model_instance):
    # type: (WordsModel) -> list
    st = model_instance.stats
    return [
        st.generate_tokens_accuracy,
        st.copy_accuracy,
        st.decision_accuracy,
    ]


def update_rules_stats(new_stats, current_stats):
    rules = new_stats
    current_stats['rules'].append(rules)


def update_words_stats(new_stats, current_stats):
    gen_tokens, copy, decision = new_stats
    current_stats['gen_tokens'].append(gen_tokens)
    current_stats['copy'].append(copy)
    current_stats['decision'].append(decision)


def get_pretrained_variables():
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='query_encoder')
    return encoder_variables
