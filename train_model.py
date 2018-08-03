import json

import batching
import model_runner
from current_net_conf import *
from model import model_full


def construct_data_sets(data_set):

    return {
        set_name: batching.construct_data_set(**{
            field_name: data_set[set_name][DATA_SET_FIELD_NAMES_MAPPING[field_name]]
            for field_name in batching.DATA_SET_FIELDS
        })
        for set_name in ('test', 'valid', 'train')
    }


def main():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    constructed_data_set = construct_data_sets(data_set)
    train_set = constructed_data_set['train']
    valid_set = constructed_data_set['valid']
    test_set = constructed_data_set['test']

    num_rule_tokens = data_set['train']['rules_tokens_count']
    num_query_tokens = data_set['train']['query_tokens_count']
    num_word_tokens = data_set['train']['words_size']

    query_end_marker = data_set['train']['query_seq_end']
    rules_end_marker = data_set['train']['rules_seq_end']
    word_end_marker = data_set['train']['words_seq_end']

    print(
        len(train_set), len(valid_set), len(test_set),
        num_rule_tokens, num_query_tokens, num_word_tokens
    )

    batcher = batching.make_batcher(query_end_marker, rules_end_marker, word_end_marker, time_major=True)
    train_batches = batching.group_by_batches(train_set, batcher, sort_key=lambda x: len(x[2]))
    valid_batches = batching.group_by_batches(valid_set, batcher, sort_key=lambda x: len(x[2]))

    model_instance = model_full.build_model(num_query_tokens, num_rule_tokens, num_word_tokens)
    model_full.apply_optimizer(model_instance)

    model_runner.train_model(train_batches, valid_batches, model_instance)


if __name__ == '__main__':
    main()
