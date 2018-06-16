import _pickle as P

import tensorflow as tf


def dump_object(obj, name):
    with open(name, 'wb') as f:
        P.dump(obj, f, protocol=2)


def load_object(name):
    with open(name, 'rb') as f:
        return P.load(f)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_object(_dict):
    return Struct(**_dict)


def print_shape(tensor, name):
    return tf.Print(tensor, [tf.shape(tensor)], name + ' ')


def fix_r_index_keys(r_index):
    return {int(key): value for key, value in r_index.items()}


def insert_words_into_placeholders(code, words, wp_token):
    word_iter = iter(words)
    result = [
        c if c != wp_token else next(word_iter, '__NO_WORD__')
        for c in code
    ]
    return result


def sequence_to_tree(sample, seq_end, subtree_start, subtree_end, word_pc):
    depth = -1
    nodes = []
    for token in sample:
        if token == seq_end:
            continue
        if token == subtree_start:
            depth += 1
        elif token == subtree_end:
            depth -= 1
        else:
            nodes.append((depth, token))

    result = []

    def get_node_on_depth(d):
        i = d
        res = result
        while i > 0:
            if not res:
                res.append(('__EMPTY_NODE__', []))
            tmp = res[-1]
            res = tmp[1]
            i -= 1
        return res

    for depth, node in nodes:
        if node == word_pc:
            depth += 1
        tree = get_node_on_depth(depth)
        tree.append((node, []))

    return result
