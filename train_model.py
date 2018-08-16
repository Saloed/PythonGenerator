import json
import argparse

import batching
import model_runner
from current_net_conf import *
from model import model_full

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['rules', 'words'],
                    help="select model to train")


def construct_data_sets(data_set, is_rules_model):
    constructor = {
        True: batching.construct_rules_data_set,
        False: batching.construct_words_data_set
    }
    return {
        set_name: constructor[is_rules_model](data_set[set_name])
        for set_name in ('test', 'valid', 'train')
    }


def main():
    args = parser.parse_args()
    is_rules_model = args.model == 'rules'

    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    num_rule_tokens = data_set['train']['rules_tokens_count']
    num_query_tokens = data_set['train']['query_tokens_count']
    num_word_tokens = data_set['train']['words_size']
    num_rule_nodes = data_set['train']['rules_nodes_count']

    query_end_marker = data_set['train']['query_seq_end']
    rules_end_marker = data_set['train']['rules_seq_end']
    word_end_marker = data_set['train']['words_seq_end']

    constructed_data_set = construct_data_sets(data_set, is_rules_model)
    train_set = constructed_data_set['train']
    valid_set = constructed_data_set['valid']
    test_set = constructed_data_set['test']

    print(
        len(train_set), len(valid_set), len(test_set),
        num_rule_tokens, num_query_tokens, num_word_tokens, num_rule_nodes
    )

    def _rules_sort_key(sample):
        return len(sample[2])  # len(rule_ids)

    def _words_sort_key(sample):
        return len(sample[-1])  # len(gen or copy)

    if is_rules_model:
        batcher = batching.make_rules_batcher(query_end_marker, rules_end_marker, time_major=True)
        sort_key = _rules_sort_key
    else:
        batcher = batching.make_words_batcher(query_end_marker, rules_end_marker, word_end_marker, time_major=True)
        sort_key = _words_sort_key

    train_batches = batching.group_by_batches(train_set, batcher, sort_key=sort_key)
    valid_batches = batching.group_by_batches(valid_set, batcher, sort_key=sort_key)

    if is_rules_model:
        model_instance = model_full.build_rules_model(num_rule_tokens, num_query_tokens, num_rule_nodes)
    else:
        model_instance = model_full.build_words_model(num_query_tokens, num_rule_tokens, num_rule_nodes,
                                                      num_word_tokens)

    model_full.apply_optimizer(model_instance, is_rules_model)
    model_runner.train_model(train_batches, valid_batches, model_instance, is_rules_model)


if __name__ == '__main__':
    main()
