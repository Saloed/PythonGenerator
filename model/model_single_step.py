from model.copy_mechanism import build_copy_mechanism_single_step
from model.encoder import build_encoder_batch_specific
from model.rules_decoder import build_rules_decoder_single_step
from model.words_encoder import build_words_encoder_with_rules
from model.words_decoder import build_words_decoder_single_step
from utils import dict_to_object, dict_plus


def build_single_step_model(query_tokens_count, rules_count, words_count):
    encoder, encoder_pc = build_encoder_batch_specific(query_tokens_count, batch_size=1)
    rules_decoder, rules_decoder_pc, rules_decoder_init = build_rules_decoder_single_step(rules_count)
    words_encoder, words_encoder_pc = build_words_encoder_with_rules(rules_count)
    words_decoder, words_decoder_pc, words_decoder_init = build_words_decoder_single_step()
    copy_mechanism, copy_mechanism_pc = build_copy_mechanism_single_step(words_decoder.words_logits, words_count)

    return dict_to_object({
        'query_encoder': dict_to_object({
            'outputs': encoder,
            'placeholders': encoder_pc
        }),

        'rules_decoder': dict_to_object({
            'outputs': rules_decoder,
            'placeholders': rules_decoder_pc,
            'initializer': rules_decoder_init
        }),
        'words_encoder': dict_to_object({
            'outputs': words_encoder,
            'placeholders': words_encoder_pc
        }),

        'words_decoder': dict_to_object({
            'outputs': dict_to_object(words_decoder.to_dict(), copy_mechanism.to_dict()),
            'placeholders': dict_plus(words_decoder_pc, copy_mechanism_pc),
            'initializer': words_decoder_init,
        })
    })
