import numpy as np
import random

from current_net_conf import *

DATA_SET_FIELDS = [
    'id',
    'query',
    'rules',
    'words',
    'words_mask',
    'copy',
    'copy_mask',
    'gen_or_copy'
]


def destruct_data_set(data_set):
    result = []
    for i, name in enumerate(DATA_SET_FIELDS):
        result.append([it[i] for it in data_set])
    return tuple(result)


def construct_data_set(**kwargs):
    if not set(kwargs.keys()) & set(DATA_SET_FIELDS):
        raise Exception('Incorrect data set')

    data_set = list(zip(*[kwargs[name] for name in DATA_SET_FIELDS]))
    return data_set


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


def make_batcher(
        query_end_marker, rules_end_marker, words_end_marker, time_major=True
):
    def preprocess_batch(batch):
        _id, query, rules, words, words_mask, copy, copy_mask, gen_or_copy = destruct_data_set(batch)
        query = align_batch(query, query_end_marker, time_major, need_length=True)
        rules = align_batch(rules, rules_end_marker, time_major, need_length=True)
        words = align_batch(words, words_end_marker, time_major, need_length=False)
        words_mask = align_batch(words_mask, 0, time_major, need_length=False)
        copy = align_batch(copy, 0, time_major, need_length=False)
        copy_mask = align_batch(copy_mask, 0, time_major, need_length=False)
        gen_or_copy = align_batch(gen_or_copy, (0, 0), time_major, need_length=True)
        return _id, query, rules, words, words_mask, copy, copy_mask, gen_or_copy

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
