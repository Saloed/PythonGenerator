import enum
import itertools
import os
import re
from typing import Iterable, Any, List

import numpy as np
import tensorflow as tf
from contracts import Tokens, Types, Decompiler
from contracts.DfsGuide import DfsGuide
from contracts.Node import Node
from contracts.Token import Token
from contracts.Tree import Tree
from contracts.TreeVisitor import TreeVisitor

from analyser import Embeddings
from analyser.Score import Score
from contants import NEXT, PAD, NOP, UNDEFINED
from logger import logger
from prepares import MAX_PARAM
from utils.Formatter import Formatter
from utils.Style import Styles
from utils.wrappers import static


def cross_entropy_loss(targets, logits, default: int = None):
    with tf.variable_scope("cross_entropy_loss"):
        if default is not None:
            with tf.variable_scope("Masking"):
                output_size = logits.get_shape()[-1].value
                default_value = tf.one_hot(default, output_size) * output_size
                boolean_mask = tf.equal(targets, -1)
                W = tf.to_int32(boolean_mask)
                targets = (1 - W) * targets + W * default
                W = tf.to_float(tf.expand_dims(W, -1))
                logits = (1 - W) * logits + W * default_value
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_sum(loss, list(range(1, len(loss.shape))))
    return loss


def l2_loss(variables):
    with tf.variable_scope("l2_loss"):
        loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return loss


def greedy_correct(targets, outputs, dependencies):
    length = max(len(output) for output in outputs)
    assert length == min(len(output) for output in outputs)
    assert length == max(len(target) for target in targets)
    assert length == min(len(target) for target in targets)
    assert length == max(len(dependency) for dependency in dependencies)
    assert length == min(len(dependency) for dependency in dependencies)
    result_targets = [None] * length
    result_dependencies = [None] * length
    from_indexes = list(range(length))
    to_indexes = list(range(length))
    for _ in range(length):
        index_best_from_index = None
        index_best_to_index = None
        best_distance = None
        for i, from_index in enumerate(from_indexes):
            for j, to_index in enumerate(to_indexes):
                distance = 0
                for target, output in zip(targets, outputs):
                    from_target = np.asarray(target[from_index])
                    to_output = np.asarray(output[to_index])
                    distance += np.linalg.norm(from_target - to_output)
                if best_distance is None or distance < best_distance:
                    index_best_from_index = i
                    index_best_to_index = j
                    best_distance = distance
        best_from_index = from_indexes[index_best_from_index]
        best_to_index = to_indexes[index_best_to_index]
        del from_indexes[index_best_from_index]
        del to_indexes[index_best_to_index]
        result_targets[best_to_index] = [target[best_from_index] for target in targets]
        result_dependencies[best_to_index] = [dependency[best_from_index] for dependency in dependencies]
    assert len(from_indexes) == 0
    assert len(to_indexes) == 0
    result_targets = [np.asarray(target) for target in zip(*result_targets)]
    result_dependencies = [np.asarray(dependency) for dependency in zip(*result_dependencies)]
    return result_targets, result_dependencies


def batch_greedy_correct(targets, outputs, dependencies):
    batch_size = max(len(output) for output in outputs)
    assert batch_size == min(len(output) for output in outputs)
    assert batch_size == max(len(target) for target in targets)
    assert batch_size == min(len(target) for target in targets)
    assert batch_size == max(len(dependency) for dependency in dependencies)
    assert batch_size == min(len(dependency) for dependency in dependencies)
    result_targets, result_dependencies = [], []
    for i in range(batch_size):
        target = [target[i] for target in targets]
        output = [output[i] for output in outputs]
        dependency = [dependency[i] for dependency in dependencies]
        target, dependency = greedy_correct(target, output, dependency)
        result_targets.append(target)
        result_dependencies.append(dependency)
    result_targets = [np.asarray(target) for target in zip(*result_targets)]
    result_dependencies = [np.asarray(dependency) for dependency in zip(*result_dependencies)]
    return result_targets, result_dependencies


def nearest_correct(targets, outputs, fine_weigh):
    tokens_targets, strings_targets, strings_mask = targets
    tokens, strings = outputs
    tokens_targets_values = np.asarray(Embeddings.tokens().idx2emb)[tokens_targets]
    result_tokens = []
    result_strings = []
    result_strings_mask = []
    batch_size = tokens.shape[0]
    contract_size = tokens.shape[1]
    for i in range(batch_size):
        default_indexes = np.arange(contract_size)
        best_indices = None
        for indexes in itertools.permutations(default_indexes):
            list_indexes = list(indexes)
            perm = tokens_targets_values[i][list_indexes]
            distance = np.linalg.norm(perm[1:] - tokens[i][1:])
            fine = fine_weigh * np.linalg.norm(default_indexes - indexes)
            distance = distance + fine
            if best_distance is None or distance < best_distance:
                best_indices = list_indexes
                best_distance = distance
        result_tokens.append(tokens_targets[i][best_indices])
        result_strings.append(strings_targets[i][best_indices])
        result_strings_mask.append(strings_mask[i][best_indices])
    result_tokens = np.asarray(result_tokens)
    result_strings = np.asarray(result_strings)
    result_strings_mask = np.asarray(result_strings_mask)
    return result_tokens, result_strings, result_strings_mask


def transpose_attention(attentions, num_heads=1):
    """
        `[a x b x c]` is tensor with shape a x b x c

        batch_size, root_time_steps, num_decoders, num_heads, num_attentions is scalars
        attn_length is array of scalars

        Input:
        attention[i] is `[root_time_steps x bach_size x attn_length[i]]`
        attention_mask is [attention[i] for i in range(num_attentions) for j in range(num_heads)]

        Output:
        attention is [`[attn_length[i]]` for i in range(num_attentions)]
        attentions is [attention for i in range(root_time_steps)]
        attention_mask is [attentions for i in range(batch_size)]
    """

    def chunks(iterable: Iterable[Any], block_size: int) -> Iterable[List[Any]]:
        result = []
        for element in iterable:
            result.append(element)
            if len(result) == block_size:
                yield result
                result = []
        if len(result) > 0:
            yield result

    # Merge multi-heading artifacts
    def merge(pack):
        return pack[0]

    attentions = [merge(attention) for attention in chunks(attentions, num_heads)]
    attentions = np.asarray(attentions)
    attentions = attentions.transpose([2, 1, 0, *range(3, len(attentions.shape))])
    return attentions


def calc_scores(labels_targets, labels,
                tokens_targets, tokens,
                strings_targets, strings,
                flatten_type):
    class Dropper(TreeVisitor):
        def visit_string(self, depth: int, node: Node, parent: Node):
            node.token = Token(Types.STRING, Types.STRING)

    def matrix_zip(depth, arrays: list) -> list:
        assert len(arrays) > 0
        if depth <= 0:
            return arrays
        length = len(arrays[0])
        get = lambda i: [array[i] for array in arrays]
        result = [matrix_zip(depth - 1, get(i)) for i in range(length)]
        return result

    def matrix_map(depth, array, mapper):
        if depth <= 0:
            return mapper(array)
        length = len(array)
        result = [matrix_map(depth - 1, array[i], mapper) for i in range(length)]
        return result

    def matrix_starmap(depth, array, mapper):
        _ = lambda x: mapper(*x)
        return matrix_map(depth, array, _)

    def matrix_parse(depth, _labels, _tokens, _strings):
        def _(label, tokens, strings) -> Tree:
            raw_tokens = []
            if tokens[0] != nop:
                for token, string in zip(tokens, strings):
                    if token == nop:
                        continue
                    token = Embeddings.tokens().get_name(token)
                    if token == Tokens.PARAM:
                        token = Tokens.PARAM + "[%d]" % MAX_PARAM
                    if token == Types.STRING:
                        string = (word for word in string if word != pad)
                        token = " ".join(Embeddings.words().get_name(word) for word in string)
                        token = '"%s"' % token.replace('"', "'")
                    raw_tokens.append(token)
            label = Embeddings.labels().get_name(label)
            if label != UNDEFINED:
                raw_tokens.insert(0, label)
            raw_tokens.insert(0, Tokens.ROOT)
            result = Decompiler.typing(raw_tokens)
            if flatten_type not in ("bfs", "dfs"):
                raise ValueError("Flatten type '%s' hasn't recognised" % flatten_type)
            try:
                if flatten_type == "bfs":
                    tree = Decompiler.bfs(result)
                if flatten_type == "dfs":
                    tree = Decompiler.dfs(result)
            except Exception:
                token = Token(Tokens.ROOT, Types.ROOT)
                root = Node(token)
                tree = Tree(root)
            return tree

        zipped = matrix_zip(depth, (_labels, _tokens, _strings))
        mapped = matrix_starmap(depth, zipped, _)
        return mapped

    def matrix_drop(depth, array):
        def _(tree: Tree):
            tree = tree.clone()
            Dropper(DfsGuide()).accept(tree)
            return tree

        mapped = matrix_map(depth, array, _)
        mapped = matrix_flatten(depth, mapped)
        return mapped

    def matrix_flatten(depth, array):
        mapped = matrix_map(depth, array, str)
        return mapped

    undefined = Embeddings.labels().get_index(UNDEFINED)
    nop = Embeddings.tokens().get_index(NOP)
    pad = Embeddings.words().get_index(PAD)
    depth = 2
    trees_targets = matrix_parse(depth, labels_targets, tokens_targets, strings_targets)
    trees = matrix_parse(depth, labels, tokens, strings)
    code_targets = matrix_flatten(depth, trees_targets)
    code = matrix_flatten(depth, trees)
    contracts_targets = matrix_drop(depth, trees_targets)
    contracts = matrix_drop(depth, trees)
    labels_scores = Score.calc(labels_targets, labels, None, undefined)
    tokens_scores = Score.calc(tokens_targets, tokens, None, nop)
    strings_scores = Score.calc(strings_targets, strings, -1, pad)
    contracts_scores = Score.calc(contracts_targets, contracts, None, Tokens.ROOT)
    code_scores = Score.calc(code_targets, code, None, Tokens.ROOT)
    return labels_scores, tokens_scores, strings_scores, contracts_scores, code_scores


def print_diff(inputs, labels_targets, labels, tokens_targets, tokens, strings_targets, strings, raw_tokens):
    class Align(enum.Enum):
        left = enum.auto()
        right = enum.auto()
        center = enum.auto()

    @static(pattern=re.compile("(\33\[\d+m)"))
    def cut(string: str, text_size: int, align: Align):
        def length(word: str):
            found = re.findall(cut.pattern, word)
            length = 0 if found is None else len("".join(found))
            return len(word) - length

        def chunks(line: str, max_line_length: int) -> Iterable[str]:
            words = line.split(" ")
            result = []
            result_length = 0
            for word in words:
                word_length = length(word)
                if result_length + word_length + 1 > max_line_length:
                    yield " ".join(result) + " " * (text_size - result_length)
                    result_length = 0
                    result = []
                result_length += word_length + 1
                result.append(word)
            yield " ".join(result) + " " * (text_size - result_length)

        lines = (" " + sub_line for line in string.split("\n") for sub_line in chunks(line, text_size - 1))
        return lines

    def print_doc(indexed_doc):
        def normalization(values: Iterable[float]) -> np.array:
            values = np.asarray(values)
            max_value = np.max(values)
            min_value = np.min(values)
            diff = (lambda diff: diff if diff > 0 else 1)(max_value - min_value)
            return (values - min_value) / diff

        def top_k_normalization(k: int, values: Iterable[float]) -> np.array:
            values = np.asarray(list(values))
            indices = np.argsort(values)[-k:]
            values = np.zeros(len(values))
            for i, j in enumerate(indices):
                values[j] = (i + 1) / k
            return values

        def split(words, *patterns):
            result = []
            for word in words:
                if any(re.findall(pattern, word) for pattern in patterns):
                    yield result
                    result = []
                else:
                    result.append(word)
            yield result

        words = (Embeddings.words().get_name(index) for index in indexed_doc)
        for text in split(words, NEXT, PAD):
            if len(text) == 0:
                continue
            for line in cut(" ".join(text), formatter.row_size(-1), Align.left):
                formatter.print("", line)

    def print_raw_tokens(raw_tokens):
        matrix = [[None for _ in range(len(raw_tokens))] for _ in range(len(Embeddings.tokens()))]
        for j, raw_token in enumerate(raw_tokens):
            color0 = lambda x: Styles.background.light_yellow if x > 1e-2 else Styles.foreground.gray
            color1 = lambda x, is_max: Styles.background.light_red if is_max else color0(x)
            color = lambda x, is_max: color1(x, is_max) % "%.3f" % x
            for i, value in enumerate(raw_token):
                matrix[i][j] = color(value, i == np.argmax(raw_token))
        for i, token in enumerate(Embeddings.tokens().idx2name):
            text = " ".join(matrix[i])
            for line in cut(text, formatter.row_size(-1), Align.left):
                formatter.print(token, line)

    def print_strings(label, tokens, strings, strings_targets):
        label = Embeddings.labels().get_name(label)
        formatter.print(label, "")
        for token, string, target in zip(tokens, strings, strings_targets):
            token = Embeddings.tokens().get_name(token)
            string = (Embeddings.words().get_name(index) if index >= 0 else " " for index in string)
            color = lambda skip: Styles.foreground.gray if skip else Styles.bold
            string = (color(index == -1) % word for word, index in zip(string, target))
            for line in cut(" ".join(string), formatter.row_size(-1), Align.left):
                formatter.print(token, line)

    formatter = Formatter(("tokens", "strings"), ("s", "s"), (20, 100))
    batch_size = len(tokens)
    for i in range(batch_size):
        num_conditions = len(tokens[i])
        formatter.print_upper_delimiter()
        print_doc(inputs[i])
        formatter.print_delimiter()
        for j in range(num_conditions):
            print_strings(labels[i][j], tokens[i][j], strings[i][j], strings_targets[i][j])
            formatter.print_delimiter()
            print_strings(labels_targets[i][j], tokens_targets[i][j], strings_targets[i][j],
                          strings_targets[i][j])
            formatter.print_delimiter()
            print_raw_tokens(raw_tokens[i][j])
            if j < num_conditions - 1:
                formatter.print_delimiter()
        formatter.print_lower_delimiter()


def print_scores(scores):
    def to_tuple(score: Score):
        f1 = score.F_score(1) * 100
        acc = score.accuracy * 100
        jcc = score.jaccard * 100
        return f1, acc, jcc

    labels_score, tokens_score, strings_score, templates_score, codes_score = scores
    formatter = Formatter(("", "F1", "Accuracy", "Jaccard"), ("s", ".1f", ".1f", ".1f"), (20, 20, 20, 20))
    formatter.raw_print = logger.error
    formatter.print_head()
    formatter.print("Labels", *to_tuple(labels_score))
    formatter.print("Tokens", *to_tuple(tokens_score))
    formatter.print("Strings", *to_tuple(strings_score))
    formatter.print("Templates", *to_tuple(templates_score))
    formatter.print("Codes", *to_tuple(codes_score))
    formatter.print_lower_delimiter()


def newest(path: str, filtrator):
    names = [path + "/" + name for name in os.listdir(path) if filtrator(path, name)]
    names.sort(key=os.path.getmtime, reverse=True)
    assert len(names) > 0, "Saves in dir '%s' hasn't found" % path
    return names[0]
