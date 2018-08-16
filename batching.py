import numpy as np
import random

from current_net_conf import *

RULES_DATA_SET_FIELDS = [
    'id',
    'query',
    'rules',
    'nodes',
    'parent_rules',
    'parent_rules_time'
]

_WORDS_DATA_SET_FIELDS = [
    'words',
    'words_mask',
    'copy',
    'copy_mask',
    'gen_or_copy'
]

WORDS_DATA_SET_FIELDS = RULES_DATA_SET_FIELDS + _WORDS_DATA_SET_FIELDS


def _destruct_data_set(data_set, fields):
    result = []
    for i, name in enumerate(fields):
        result.append([it[i] for it in data_set])
    return tuple(result)


def destruct_rules_data_set(data_set):
    return _destruct_data_set(data_set, RULES_DATA_SET_FIELDS)


def destruct_words_data_set(data_set):
    return _destruct_data_set(data_set, WORDS_DATA_SET_FIELDS)


def align_batch(batch, padding_data, time_major=True, need_length=True):
    lens = [len(b) for b in batch]
    max_len = max(lens)
    result = np.asarray([
        sample + [padding_data for _ in range(max_len - sample_len)]
        for sample, sample_len in zip(batch, lens)
    ])
    if time_major:
        current_shape = np.shape(result)
        axes = list(range(len(current_shape)))
        transposed_axes = [1, 0] + axes[2:]
        result = result.transpose(transposed_axes)

    if not need_length:
        return result

    return result, np.asarray(lens)


def make_rules_batcher(
        query_end_marker, rules_end_marker, time_major=True
):
    def preprocess_batch(batch):
        _id, query, rules, nodes, parent_rules, parent_rules_t = destruct_rules_data_set(batch)
        query = align_batch(query, query_end_marker, time_major, need_length=True)
        rules = align_batch(rules, rules_end_marker, time_major, need_length=True)
        nodes = align_batch(nodes, -1, time_major, need_length=False)
        parent_rules = align_batch(parent_rules, -1, time_major, need_length=False)
        parent_rules_t = align_batch(parent_rules_t, 0, time_major, need_length=False)
        return _id, query, rules, nodes, parent_rules, parent_rules_t

    return preprocess_batch


def make_words_batcher(
        query_end_marker, rules_end_marker, words_end_marker, time_major=True
):
    rules_batch_preprocessor = make_rules_batcher(query_end_marker, rules_end_marker, time_major)

    def preprocess_batch(batch):
        rules_part = [b[:len(RULES_DATA_SET_FIELDS)] for b in batch]
        words_part = [b[len(RULES_DATA_SET_FIELDS):] for b in batch]
        _id, query, rules, nodes, parent_rules, parent_rules_t = rules_batch_preprocessor(rules_part)
        words, words_mask, copy, copy_mask, gen_or_copy = _destruct_data_set(words_part, _WORDS_DATA_SET_FIELDS)
        words = align_batch(words, words_end_marker, time_major, need_length=False)
        words_mask = align_batch(words_mask, 0, time_major, need_length=False)
        copy = align_batch(copy, 0, time_major, need_length=False)
        copy_mask = align_batch(copy_mask, 0, time_major, need_length=False)
        gen_or_copy = align_batch(gen_or_copy, (0, 0), time_major, need_length=True)
        return _id, query, rules, nodes, parent_rules, parent_rules_t, words, words_mask, copy, copy_mask, gen_or_copy

    return preprocess_batch


def group_by_batches(data_set, batch_preprocess_fn, shuffle=True, sort_key=None):
    batch_count = len(data_set) // BATCH_SIZE
    batches = []
    if sort_key is not None:
        data_set.sort(key=sort_key)
    for j in range(batch_count):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        processed = batch_preprocess_fn(d)
        batches.append(processed)
    if shuffle:
        random.shuffle(batches)
    return batches


def construct_rules_data_set(data_set, rules_field_name='rules_data'):
    rules_data_set = {name: [] for name in RULES_DATA_SET_FIELDS}
    for _id, query, rules_d in zip(data_set['ids'], data_set['queries'], data_set[rules_field_name]):
        rules_data_set['id'].append(_id)
        rules_data_set['query'].append(query)
        rule_ids, parent_info = rules_d
        node_ids, parent_rule_ids, parent_rule_times = parent_info
        rules_data_set['rules'].append(rule_ids)
        rules_data_set['nodes'].append(node_ids)
        rules_data_set['parent_rules'].append(parent_rule_ids)
        rules_data_set['parent_rules_time'].append(parent_rule_times)

    data_set = list(zip(*[rules_data_set[name] for name in RULES_DATA_SET_FIELDS]))
    return data_set


def construct_words_data_set(data_set):
    rules_data_set = construct_rules_data_set(data_set, rules_field_name='rules_data_with_placeholders')
    words_data_set = {name: [] for name in _WORDS_DATA_SET_FIELDS}

    for words_data in data_set['words_data']:
        _words_data, parent_node_info = words_data
        word_ids, word_mask, copy_ids, copy_mask, generate_or_copy = _words_data
        words_data_set['words'].append(word_ids)
        words_data_set['words_mask'].append(word_mask)
        words_data_set['copy'].append(copy_ids)
        words_data_set['copy_mask'].append(copy_mask)
        words_data_set['gen_or_copy'].append(generate_or_copy)

    _words_data_set = list(zip(*[words_data_set[name] for name in _WORDS_DATA_SET_FIELDS]))

    data_set = [rds + wds for rds, wds in zip(rules_data_set, _words_data_set)]
    return data_set
