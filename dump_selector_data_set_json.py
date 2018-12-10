from __future__ import division
from __future__ import print_function

import json
import pickle
import random

import tensorflow as tf
from tqdm import tqdm

import batching
from current_net_conf import *
from evaluation.generate_code import CodeGenerator
from model.model_single_step import build_single_step_rules_model
from model.model_single_step import get_rules_variables

DATA_PER_SAMPLE = 5


def extract_sequence_from_rule_tree(rule_tree, rules_end, nodes_end):
    rules, nodes, parent_rules = rule_tree.make_sequence()
    rules.append(rules_end)
    nodes.append(nodes_end)
    parent_rules.append(rules_end)
    return rules, nodes, parent_rules


def make_data(sample_id, query, generated_data, true_data):
    for _ in range(DATA_PER_SAMPLE):
        data = generated_data[:]
        random.shuffle(data)
        true_data_id = data.index(true_data)
        yield sample_id, query, true_data_id, data


def generate_selector_data_for_sample(sample, code_generator, end_markers):
    rules_end = end_markers['rules']
    nodes_end = end_markers['nodes']

    sample_id, query, rules, nodes, parent_rules, _ = sample

    rules[-1] = rules_end
    nodes[-1] = nodes_end
    parent_rules[-1] = rules_end

    true_data = rules, nodes, parent_rules

    sample_rule_trees = code_generator.generate_rules_for_query(query, None)
    generated_data = [
        extract_sequence_from_rule_tree(rule_tree, rules_end, nodes_end)
        for rule_tree in sample_rule_trees
    ]

    if true_data not in generated_data:
        generated_data.append(true_data)

    return make_data(sample_id, query, generated_data, true_data)


def generate_selector_data_set(data_set, code_generator, end_markers):
    return [
        data
        for sample in tqdm(data_set)
        for data in generate_selector_data_for_sample(sample, code_generator, end_markers)
    ]


def generate():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    with open(DATA_SET_BASE_DIR + FULL_DATA_SET_NAME) as f:
        full_data_set = pickle.load(f)

    full_train_set, _, _ = full_data_set

    num_rule_tokens = data_set['train']['rules_tokens_count']
    num_query_tokens = data_set['train']['query_tokens_count']
    num_rule_nodes = data_set['train']['rules_nodes_count']

    query_end_marker = data_set['train']['query_seq_end']
    rules_end_marker = data_set['train']['rules_seq_end']

    rules_model = build_single_step_rules_model(num_query_tokens, num_rule_tokens, num_rule_nodes)

    grammar = full_train_set.grammar

    counts = {
        'query': num_query_tokens,
        'rules': num_rule_tokens,
        'nodes': num_rule_nodes
    }

    end_markers = {
        'query': query_end_marker,
        'rules': rules_end_marker,
    }

    result = {}

    rules_model_saver = tf.train.Saver(get_rules_variables())
    rules_model_name = MODEL_SAVE_PATH + BEST_RULES_MODEL_BASE_NAME

    with tf.Session() as session:
        rules_model_saver.restore(session, rules_model_name)

        code_generator = CodeGenerator(
            session=session,
            rules_model=rules_model,
            words_model=None,
            rules_grammar=grammar,
            words_r_index=None,
            seq_end_markers=end_markers,
            counts=counts,
            rules_word_placeholder=None
        )

        end_markers['nodes'] = counts['nodes'] + 1
        counts['nodes'] += 2

        for set_name in ['valid', 'train', 'test']:
            sample_subset = batching.construct_rules_data_set(data_set[set_name])
            selector_data_set = generate_selector_data_set(sample_subset, code_generator, end_markers)
            result[set_name] = {}
            result[set_name]['data'] = selector_data_set
            for name, value in end_markers.items():
                result[set_name][name + '_end_marker'] = value
            for name, value in counts.items():
                result[set_name][name + '_count'] = value

    with open('tmp/selector_ds', 'wb') as f:
        pickle.dump(result, f)

    with open(DATA_SET_BASE_DIR + DATA_SET_NAME + '_selector', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    generate()
