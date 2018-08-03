import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.copy_mechanism import build_copy_mechanism, build_words_loss
from model.encoder import build_encoder
from model.rules_decoder import build_rules_decoder, build_rules_loss
from model.words_decoder import build_words_decoder
from model.words_encoder import build_words_encoder
from utils import dict_to_object


def build_model(query_tokens_count, rules_count, words_count):
    query_encoder, encoder_placeholders = build_encoder(query_tokens_count)
    rules_decoder, rules_decoder_placeholders = build_rules_decoder(query_encoder.last_state, rules_count)
    words_encoder, words_placeholders = build_words_encoder(rules_count, rules_decoder_placeholders)
    words_decoder, words_decoder_pc = build_words_decoder(words_encoder.last_state)
    copy_mechanism = build_copy_mechanism(query_encoder.all_states, words_decoder.all_states, words_count)

    with variable_scope('loss'):
        rules_loss, rules_stats = build_rules_loss(rules_decoder)
        words_loss, words_stats, words_placeholders = build_words_loss(copy_mechanism, words_decoder_pc)
        all_loss = list(rules_loss.values()) + list(words_loss.values())
        stacked_loss = tf.stack(all_loss)
        loss = tf.reduce_sum(stacked_loss)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        l2_variables = [v for v in variables if v.name.split('/')[-1].startswith('kernel')]
        l2_losses = [tf.nn.l2_loss(v) for v in l2_variables]
        l2_loss = L2_COEFFICIENT * tf.reduce_sum(l2_losses)
        loss_with_l2 = loss + l2_loss

    placeholders = dict_to_object(
        encoder_placeholders,
        rules_decoder_placeholders,
        words_placeholders,
        words_decoder_pc,
    )

    stats = dict_to_object(
        rules_stats,
        words_stats
    )

    return dict_to_object({
        'placeholders': placeholders,
        'loss': loss,
        'loss_with_l2': loss_with_l2,
        'stats': stats
    })


def apply_optimizer(model_instance):
    optimizer = tf.train.AdamOptimizer(0.0005)
    updates = optimizer.minimize(model_instance.loss_with_l2)
    model_instance.updates = updates


def make_feed(data, model_instance):
    (q_ids, q_length), (rules, rules_length), words, words_mask, copy, copy_mask, (gen_or_copy, words_length) = data
    pc = model_instance.placeholders
    return {
        pc.query_ids: q_ids,
        pc.query_length: q_length,
        pc.rules_sequence_length: rules_length,
        pc.rules_target: rules,
        pc.generate_token_target: words,
        pc.generate_token_target_mask: words_mask,
        pc.copy_target: copy,
        pc.copy_target_mask: copy_mask,
        pc.generate_or_copy_target: gen_or_copy,
        pc.words_sequence_length: words_length,
    }


def make_fetches(model_instance, is_train):
    fetches = [model_instance.loss]
    if is_train:
        fetches += [model_instance.updates]
    return fetches


def make_stats_fetches(model_instance):
    st = model_instance.stats
    return [
        st.rules_accuracy,
        st.generate_tokens_accuracy,
        st.copy_accuracy,
        st.decision_accuracy,
    ]


def update_stats(new_stats, current_stats):
    rules, gen_tokens, copy, decision = new_stats
    current_stats['rules'].append(rules)
    current_stats['gen_tokens'].append(gen_tokens)
    current_stats['copy'].append(copy)
    current_stats['decision'].append(decision)


def get_pretrained_variables():
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='query_encoder')
    return encoder_variables
