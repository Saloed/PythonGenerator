from __future__ import print_function
from __future__ import division

import six
import json
import pickle

from current_net_conf import DATA_SET_BASE_DIR, DATA_SET_NAME, FULL_DATA_SET_NAME
from evaluation.structures import RulesTree

SEQUENCE_END_TOKEN = '___SEQ_END___'


def make_tree_token_index(trees):
    class Context:
        token_set = set()

    def tokens(node):
        # global _tokens
        node, children = node
        Context.token_set |= {node}
        for child in children:
            tokens(child)

    for tree in trees:
        tokens(tree)

    return {tk: i for i, tk in enumerate(sorted(Context.token_set))}


def indexate_trees(trees, index):
    def indexate_tree(root):
        node, children = root
        node = index[node]
        children = [indexate_tree(child) for child in children]
        return node, children

    return [indexate_tree(tree) for tree in trees]


def find_placeholders(rules_data, grammar, rule_pc_id):
    rule_ids, _ = rules_data

    rule_tree = RulesTree.create_new(grammar)
    for rule_id in rule_ids:
        rule = grammar.id_to_rule[rule_id]
        rule_tree = rule_tree.apply(rule, 0)

    rules_with_pc, node_ids_pc, p_rule_ids_pc = rule_tree.make_sequence_with_placeholders(rule_pc_id)
    p_rule_times_pc = [None for _ in p_rule_ids_pc]
    node_parent_data_pc = node_ids_pc, p_rule_ids_pc, p_rule_times_pc
    return rules_with_pc, node_parent_data_pc


def add_rules_data_end_markers(rules_data, rules_end_id):
    rule_ids, node_parent_data = rules_data
    rule_ids.append(rules_end_id)
    for npd in node_parent_data:
        npd.append(npd[-1])
    return rule_ids, node_parent_data


def add_words_data_end_markers(words_data, words_end_id):
    words_data, node_parent_data = words_data
    word_ids, word_mask, copy_ids, copy_mask, generate_or_copy = words_data

    word_ids.append(words_end_id)
    word_mask.append(1)
    copy_ids.append(0)
    copy_mask.append(0)
    generate_or_copy.append((0, 1))

    for npd in node_parent_data:
        npd.append(npd[-1] if npd else -1)

    words_data = word_ids, word_mask, copy_ids, copy_mask, generate_or_copy
    return words_data, node_parent_data


def rework(data_set, indexes, grammar):
    ids = []
    trees = []
    queries = []
    all_rules_data = []
    all_words_data = []
    all_rules_data_with_pc = []

    samples = data_set.examples

    annot_end_id = indexes['query_seq_end']
    rules_end_id = indexes['rules_seq_end']
    words_end_id = indexes['words_seq_end']
    rules_pc_id = indexes['rules_word_pc']

    for sample in samples:
        sample_data = sample.data
        sample_id = sample.raw_id
        sample_tree = get_example_tree(sample)
        query_ids, rules_data, words_data = sample_data

        rules_data_with_placeholders = find_placeholders(rules_data, grammar, rules_pc_id)

        query_ids.append(annot_end_id)

        rules_data = add_rules_data_end_markers(rules_data, rules_end_id)
        rules_data_with_placeholders = add_rules_data_end_markers(rules_data_with_placeholders, rules_end_id)

        words_data = add_words_data_end_markers(words_data, words_end_id)

        ids.append(sample_id)
        trees.append(sample_tree)
        queries.append(query_ids)

        all_rules_data.append(rules_data)
        all_rules_data_with_pc.append(rules_data_with_placeholders)
        all_words_data.append(words_data)

    tree_index = make_tree_token_index(trees)
    trees = indexate_trees(trees, tree_index)
    tree_size = len(tree_index)

    data_set = {
        'queries': queries,

        'rules_data': all_rules_data,
        'rules_data_with_placeholders': all_rules_data_with_pc,
        'words_data': all_words_data,

        'trees': trees,
        'tree_size': tree_size,

        'ids': ids
    }

    data_set.update(indexes)

    return data_set


def extract_rules_data(ex_data):
    rule_ids = []
    node_ids = []
    parent_rule_ids = []
    parent_rule_times = []
    time = 0
    time_mapping = {}
    for i, ((rule_id, _, _), (is_rule, _, _), node_id, parent_rule_id, parent_rule_t) in enumerate(zip(*ex_data)):
        if not int(is_rule):
            continue
        time_mapping[i] = time
        time += 1
        rule_ids.append(int(rule_id))
        node_ids.append(int(node_id))
        parent_rule_ids.append(int(parent_rule_id))
        parent_rule_times.append(int(parent_rule_t))

    parent_rule_times = [time_mapping[t] for t in parent_rule_times]
    node_parent_data = node_ids, parent_rule_ids, parent_rule_times
    return rule_ids, node_parent_data


def extract_words_data(ex_data, terminal_vocab):
    word_ids, word_mask = [], []
    copy_ids, copy_mask = [], []
    generate_or_copy = []

    node_ids = []
    parent_rule_ids = []
    parent_rule_times = []

    for (_, word_id, copy_position), (is_r, is_word, is_copy), node_id, parent_rule_id, parent_rule_t in zip(*ex_data):
        if is_r:
            continue
        word_id, copy_position = int(word_id), int(copy_position)
        is_word, is_copy = int(is_word), int(is_copy)

        copy_ids.append(copy_position if is_copy else 0)
        word_ids.append(word_id if is_word else 0)

        copy_mask.append(is_copy)
        word_mask.append(is_word)

        if is_copy and is_word:
            gen_or_copy = (1, 0.75)
        else:
            gen_or_copy = (is_copy, is_word)

        generate_or_copy.append(gen_or_copy)

        node_ids.append(int(node_id))
        parent_rule_ids.append(int(parent_rule_id))
        parent_rule_times.append(int(parent_rule_t))

    if word_ids and word_ids[-1] != terminal_vocab.eos:
        word_ids.append(terminal_vocab.eos)
        word_mask.append(1)
        copy_ids.append(0)
        copy_mask.append(0)
        generate_or_copy.append((0, 1))
        node_ids.append(node_ids[-1])
        parent_rule_ids.append(parent_rule_ids[-1])
        parent_rule_times.append(parent_rule_times[-1])

    words_data = word_ids, word_mask, copy_ids, copy_mask, generate_or_copy
    node_parent_data = node_ids, parent_rule_ids, parent_rule_times
    return words_data, node_parent_data


def remake_sample(example, terminal_vocab):
    ex_data = [it[0] for it in example.data]
    tar_ex_data = ex_data[1:]
    rules_data = extract_rules_data(tar_ex_data)
    words_data = extract_words_data(tar_ex_data, terminal_vocab)
    query_ids = [int(i) for i in ex_data[0]]
    data = query_ids, rules_data, words_data
    example._data = data


def remake_data_set(data_set, terminal_vocab):
    for i, sample in enumerate(data_set):
        remake_sample(sample, terminal_vocab)


def get_tree(root):
    children = [get_tree(child) for child in root.children]
    node = root.type
    node = node if isinstance(node, six.string_types) else node.__name__
    return node, children


def get_example_tree(example):
    return get_tree(example.parse_tree)


def huyalisis(sample):
    ex_data = [it[0] for it in sample.data]

    data = list(ex_data[1])
    is_data = list(ex_data[2])

    assert is_data[0][0] == 1, sample.raw_id


def make_huyalisis(samples):
    for sample in samples:
        huyalisis(sample)


def prepare_for_dump(data_set):
    terminal_vocab = data_set.terminal_vocab
    data_set.init_data_matrices()
    samples = data_set.examples
    remake_data_set(samples, terminal_vocab)
    return data_set


def extract_indexes_from_data_set(ds):
    annot_index = ds.annot_vocab.token_id_map
    words_index = ds.terminal_vocab.token_id_map
    rules_count = len(ds.grammar.id_to_rule)
    nodes_count = len(ds.grammar.node_type_to_id)

    return {
        'query_index': annot_index,
        'rules_tokens_count': rules_count,
        'words_index': words_index,
        'nodes_count': nodes_count,
    }


def modify_indexes(data_sets):
    data_set_indexes = [extract_indexes_from_data_set(ds) for ds in data_sets]
    for dsi in data_set_indexes[1:]:
        assert dsi == data_set_indexes[0]

    index = data_set_indexes[0]

    annot_index = index['query_index']
    annot_end_id = len(annot_index)
    annot_index[SEQUENCE_END_TOKEN] = annot_end_id

    words_index = index['words_index']
    words_end_id = len(words_index)
    words_index[SEQUENCE_END_TOKEN] = words_end_id

    rules_count = index['rules_tokens_count']
    rules_end_id = rules_count + 0
    rules_pc_id = rules_count + 1
    rules_count += 2

    nodes_count = index['nodes_count']

    words_count = len(words_index)
    annot_count = len(annot_index)

    return {
        'query_index': annot_index,
        'query_seq_end': annot_end_id,
        'query_tokens_count': annot_count,

        'rules_seq_end': rules_end_id,
        'rules_word_pc': rules_pc_id,
        'rules_tokens_count': rules_count,

        'rules_nodes_count': nodes_count,

        'words_index': words_index,
        'words_seq_end': words_end_id,
        'words_size': words_count,
    }


def main():
    with open(DATA_SET_BASE_DIR + FULL_DATA_SET_NAME, 'rb') as f:
        data_sets = pickle.load(f)

    prepared_data_sets = [prepare_for_dump(ds) for ds in data_sets]
    new_indexes = modify_indexes(prepared_data_sets)
    reworked_data_sets = [rework(ds, new_indexes, data_sets[2].grammar) for ds in prepared_data_sets]
    train, dev, test = reworked_data_sets

    result = {
        'train': train,
        'valid': dev,
        'test': test,
    }
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
