import tensorflow as tf

from model.copy_mechanism import build_copy_mechanism_single_step, CopyMechanism, CopyMechanismPlaceholdersSingleStep
from model.encoder import QueryEncoderPlaceholders, QueryEncoder
from model.encoder import build_query_encoder_for_rules, build_query_encoder_for_words
from model.rules_decoder import RulesDecoderPlaceholdersSingleStep, RulesDecoderSingleStep
from model.rules_decoder import build_rules_decoder_single_step
from model.selector import Selector
from model.selector import SelectorPlaceholders
from model.selector import build_selector
from model.words_decoder import WordsDecoderSingleStep, WordsDecoderPlaceholdersSingleStep
from model.words_decoder import build_words_decoder_single_step
from model.words_encoder import WordsEncoder
from model.words_encoder import WordsEncoderPlaceholders
from model.words_encoder import build_words_encoder
from utils import Magic


class _QueryEncoder:
    def __init__(self, outputs, placeholders):
        self.outputs = outputs  # type: QueryEncoder
        self.placeholders = placeholders  # type: QueryEncoderPlaceholders


class _RulesDecoder:
    def __init__(self, outputs, placeholders, initializer):
        self.outputs = outputs  # type: RulesDecoderSingleStep
        self.placeholders = placeholders  # type: RulesDecoderPlaceholdersSingleStep
        self.initializer = initializer


class _WordsEncoder:
    def __init__(self, outputs, placeholders):
        # type: (WordsEncoder, WordsEncoderPlaceholders) -> None
        self.outputs = outputs
        self.placeholders = placeholders


class _WordsDecoderOutputs(Magic, WordsDecoderSingleStep, CopyMechanism):
    pass


class _WordsDecoderPlaceholders(Magic, WordsDecoderPlaceholdersSingleStep, CopyMechanismPlaceholdersSingleStep):
    pass


class _WordsDecoder:
    def __init__(self, outputs, placeholders, initializer):
        self.outputs = outputs  # type: _WordsDecoderOutputs
        self.placeholders = placeholders  # type: _WordsDecoderPlaceholders
        self.initializer = initializer


class RulesModelSingleStep:
    def __init__(self, query_encoder, rules_decoder):
        # type: (_QueryEncoder, _RulesDecoder) -> None
        self.query_encoder = query_encoder
        self.rules_decoder = rules_decoder


class WordsModelSingleStep:
    def __init__(self, query_encoder, words_encoder, words_decoder):
        self.query_encoder = query_encoder  # type: _QueryEncoder
        self.words_encoder = words_encoder  # type: _WordsEncoder
        self.words_decoder = words_decoder  # type: _WordsDecoder


class SelectorModel:
    def __init__(self, outputs, placeholders):
        self.outputs = outputs  # type: Selector
        self.placeholders = placeholders  # type: SelectorPlaceholders


def build_single_step_rules_model(query_tokens_count, rules_count, nodes_count):
    encoder, encoder_pc = build_query_encoder_for_rules(query_tokens_count, batch_size=1)
    rules_decoder, rules_decoder_pc, rules_decoder_init = build_rules_decoder_single_step(rules_count, nodes_count)
    query_encoder = _QueryEncoder(encoder, encoder_pc)
    rules_decoder = _RulesDecoder(rules_decoder, rules_decoder_pc, rules_decoder_init)
    model = RulesModelSingleStep(query_encoder, rules_decoder)
    return model


def get_rules_variables():
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='query_encoder')
    decoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rules_decoder')
    return encoder_variables + decoder_variables


def build_single_step_words_model(query_tokens_count, rules_count, nodes_count, words_count):
    encoder, encoder_pc = build_query_encoder_for_words(query_tokens_count, batch_size=1)
    words_encoder, words_encoder_pc = build_words_encoder(rules_count, nodes_count, batch_size=1)
    words_decoder, words_decoder_pc, words_decoder_init = build_words_decoder_single_step()
    copy_mechanism, copy_mechanism_pc = build_copy_mechanism_single_step(words_decoder.words_logits, words_count)

    query_encoder = _QueryEncoder(encoder, encoder_pc)
    words_encoder = _WordsEncoder(words_encoder, words_encoder_pc)

    words_decoder_outputs = _WordsDecoderOutputs(words_decoder, copy_mechanism)
    words_decoder_placeholders = _WordsDecoderPlaceholders(words_decoder_pc, copy_mechanism_pc)
    words_decoder = _WordsDecoder(words_decoder_outputs, words_decoder_placeholders, words_decoder_init)

    model = WordsModelSingleStep(query_encoder, words_encoder, words_decoder)
    return model


def get_words_variables():
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='words_query_encoder')
    words_encoder = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='words_encoder')
    words_decoder = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='words_decoder')
    copy_mechanism = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='copy_mechanism')
    return encoder_variables + words_encoder + words_decoder + copy_mechanism


def build_selector_model(query_tokens_count, rules_count, nodes_count):
    selector, selector_pc = build_selector(query_tokens_count, rules_count, nodes_count)
    return SelectorModel(selector, selector_pc)


def get_selector_variables():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='selector')
