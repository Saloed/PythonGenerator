import functools
import sys
import ast
import html
import re
import _pickle as P

import timeout_decorator as td
import pandas as pd
import numpy as np
import tensorflow as tf

import DummySeq2Seq
from hlam import tree_transformers

from collections import defaultdict
from traceback import print_tb
from multiprocessing import Pool

from hlam.tree_transformers import CONSTANT_LITERAL_TYPE


# from some_net_stuff import Seq2Seq


@td.timeout(10)
def parse_body(body):
    codez = re.finditer(r"<p>(.*)</p>\s+<pre[^>]*>[^<]*<code[^>]*>((?:\s|[^<]|<span[^>]*>[^<]+</span>)*)</code></pre>",
                        body)
    codez = map(lambda x: (x.group(1), x.group(2)), codez)
    for message, code in sorted(codez, key=lambda x: len(x), reverse=True):
        # fetch that code
        code = html.unescape(code)
        code = re.sub(r"<[^>]+>([^<]*)<[^>]*>", "\1", code)
        try:
            ast.parse(code)
            return message, code
        except:
            pass
    return None, None


def parse_question_with_answer(qanda):
    q, a = qanda
    try:
        q = parse_body(q)
        a = parse_body(a)
    except Exception:
        return (None, None), (None, None)
    return q, a


def parse(name):
    data = pd.read_csv(name)
    data_to_parse = list(sorted(zip(data['question'], data['answer']), key=lambda x: len(x[0]) + len(x[1])))
    num_tasks = len(data_to_parse)
    parsed = []
    with Pool(8) as pool:
        for i, result in enumerate(pool.imap_unordered(parse_question_with_answer, data_to_parse), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_tasks))
            parsed.append(result)
    parsed = [(q_text, q_code, a_text, a_code) for ((q_text, q_code), (a_text, a_code)) in parsed]
    parsed = pd.DataFrame(data=parsed, columns=['question_text', 'question_code', 'answer_text', 'answer_code'])
    parsed.to_csv('ParsedData.csv')


def transform_tree(tree, text):
    # try:
    return tree_transformers.NodeUebator().visit(tree)
    # except Exception as ex:
    #     print(ex)
    #     # print(text)
    #     return None


def extract_common_names(root, text):
    # ids = {}
    # names = {}
    # def extract_fields(node):
    #     result = {'node': node.__class__.__name__, 'fields': {}}
    #     for field_name, value in ast.iter_fields(node):
    #         if field_name in {'id', 'name', 'arg', 'ctx'}:
    #             if field_name == 'id':
    #                 if value not in ids:
    #                     ids[value] = max(ids.values()) + 1
    #                 value = ids[value]
    #             elif field_name == 'ctx':
    #                 value = value.__class__.__name__
    #             result['fields'][field_name] = value
    #     return result
    try:
        attribute_names, function_names = {}, {}
        tree_transformers.set_parents_in_tree(root)
        node = tree_transformers.NodePereebator(attribute_names, function_names).visit(root)
    except Exception as ex:
        print(ex)
        return None
    return node, (attribute_names, function_names)


def replace_names(root, attribute_names, function_names):
    return tree_transformers.NodePerehuyator(attribute_names, function_names).visit(root)


def count_items(_list, sort=False):
    items_with_count = defaultdict(int)
    for it in _list:
        items_with_count[it] += 1
    items_with_count = list(items_with_count.items())
    if sort:
        items_with_count.sort(key=lambda it: it[1], reverse=True)
    return items_with_count


def get_most_common_names(all_names):
    common_names = count_items(all_names, sort=True)
    return {name for name, count in common_names[:100]}


def get_constant_literals(node, literals=None):
    if literals is None:
        literals = []
    if not isinstance(node, dict):
        return literals
    if node['type'] == tree_transformers.CONSTANT_LITERAL_TYPE:
        literals.append(node['value'])
        return literals
    for field, value in node.items():
        if field == 'type':
            continue
        if not isinstance(value, list):
            value = [value]
        for it in value:
            get_constant_literals(it, literals)
    return literals


def tree_to_token_stream(tree):
    pass


def children_as_list(tree):
    if not isinstance(tree, dict):
        return tree
    if tree['type'] in [CONSTANT_LITERAL_TYPE, tree_transformers.EMPTY_TOKEN]:
        return tree
    children = []
    for name, node in tree.items():
        if name == 'type':
            continue
        if not isinstance(node, list):
            node = [node]
        children += [children_as_list(it) for it in node]

    return {
        'type': tree['type'],
        'children': children,
    }


def replace_empty_tokens(tree):
    if not isinstance(tree, dict):
        return tree
    for name, node in tree.items():
        if node is None:
            tree[name] = tree_transformers.make_empty_token()
            continue
        if not isinstance(node, list):
            node = [node]
        for it in node:
            replace_empty_tokens(it)
    return tree


def find_non_terminals(tree, parent=None):
    if not isinstance(tree, dict):
        return [parent]
    result = []
    if tree['type'] == CONSTANT_LITERAL_TYPE:
        return result
    for name, node in tree.items():
        if name == 'type':
            continue
        if isinstance(node, list):
            result += [x for it in node for x in find_non_terminals(it, tree)]
        else:
            result += find_non_terminals(node, tree)

    return result


def extract_tokens_from_tree(tree):
    return [
               tree['type']
           ] + [
               token
               for child in tree.get('children', [])
               for token in extract_tokens_from_tree(child)
           ]


def extract_tokens(trees):
    tokens = [extract_tokens_from_tree(tree) for tree in trees]
    result = set()
    for t in tokens:
        result |= set(t)
    result = list(result)
    result.sort()
    return result


def split_batches(batches, split_size):
    split_position = len(batches) // split_size
    return batches[:split_position], batches[split_position:]


# def run_with_outputs(
#         data_set,
#         model,
#         session,
#         num_text_tokens_with_end,
#         num_code_tokens_with_end,
# ):
#     inputs, input_length, target_labels, target_length, outputs, loss = model
#     res = []
#     for ((text, text_lens), (code, code_lens)) in data_set:
#         # text, text_embs = get_sample_embs(text, num_text_tokens_with_end)
#         # code, code_embs = get_sample_embs(code, num_code_tokens_with_end)
#         feed = {
#             inputs: text,
#             input_length: text_lens,
#             target_labels: code,
#             target_length: code_lens,
#         }
#         output = session.run(fetches=outputs, feed_dict=feed)
#         res.append((output, code))
#     return res


# def fix_results(results):
#     res = []
#
#     def flat(smth):
#         flatten = [[] for _ in range(len(smth[0]))]
#         for batch_at_time in smth:
#             for i, it in enumerate(batch_at_time):
#                 flatten[i].append(it)
#         return flatten
#
#     for outputs, targets in results:
#         outputs = flat(outputs)
#         targets = flat(targets)
#         res += list(zip(outputs, targets))
#     return res


# def seq2seq_text2code_results(text, code, num_text_tokens, num_code_tokens):
#     batch_size = 1
#     num_text_tokens_with_end = num_text_tokens + 2
#     num_code_tokens_with_end = num_code_tokens + 2
#
#     # model = Seq2Seq.build_model(
#     #     batch_size,
#     #     num_text_tokens_with_end,
#     #     num_code_tokens_with_end,
#     #     is_in_train_mode=True,
#     # )
#
#     model = spizjenno_runner.run(num_text_tokens_with_end, num_code_tokens_with_end, batch_size)
#     batches = group_text_and_code_by_batches(text, code, batch_size, num_text_tokens_with_end,
#       num_code_tokens_with_end)
#     valid, train = split_batches(batches, 10)
#
#     config = tf.ConfigProto()
#     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#
#     saver = tf.train.Saver(max_to_keep=100)
#     with tf.Session(config=config) as sess, tf.device('/cpu:0'):
#         saver.restore(sess, 'models/model-0')
#         results = run_with_outputs(valid, model, sess, num_text_tokens_with_end, num_code_tokens_with_end)
#         results = fix_results(results)
#         with open('some_results.dmp', 'wb') as f:
#             P.dump(results, f)


def some_analysis(r_index):
    index_size = len(r_index) + 2
    end_marker_index = index_size - 1
    start_marker_index = index_size - 2

    r_index[end_marker_index] = '__EndMarker__'
    r_index[start_marker_index] = '__StartMarker__'

    with open('some_results.dmp', 'rb') as f:
        results = P.load(f)
    words = [
        (
            [r_index[it] for it in outputs],
            [r_index[it] for it in targets]
        ) for outputs, targets in results
    ]
    wwww = words


def get_word_embedding(word):
    if isinstance(word, list):
        coeff = 1 / len(word)
        return [(ch, coeff) for ch in word]
    return [(word, 1.0)]


def make_feed_from_data_set(data_set, input_ids, input_weights, input_length, target_labels, target_length):
    for (text, text_lens), (code, code_lens) in data_set:
        text_embs_with_weight = [
            [
                emb_with_weight
                for word in sample
                for emb_with_weight in get_word_embedding(word)
            ]
            for sample in text
        ]

        text_embs = [[emb for emb, _ in sample] for sample in text_embs_with_weight]
        text_weights = [[[w] for _, w in sample] for sample in text_embs_with_weight]

        text_inputs = np.asarray(text_embs).transpose([1, 0])
        text_weights = np.asarray(text_weights).transpose([1, 0, 2])
        code_inputs = np.asarray(code).transpose([1, 0])
        yield {
            input_ids: text_inputs,
            input_weights: text_weights,
            input_length: text_lens,
            target_labels: code_inputs,
            target_length: code_lens,
        }


def build_loss_summary(model):
    *_, loss = model
    tf.summary.scalar('loss', loss)
    return tf.summary.merge_all()


def build_updates(model):
    *_, loss = model
    return tf.train.RMSPropOptimizer(0.005).minimize(loss)
    # return tf.train.AdamOptimizer().minimize(loss)


def run_seq2seq_text_model(
        data_set, model, session, epoch, is_train,
        updates=None, summary_writer=None, summaries=None
):
    input_ids, input_weights, input_length, target_labels, target_length, outputs, loss = model
    fetches = [loss]
    if is_train:
        fetches += [summaries, updates]
    res = []
    batch_count = len(data_set)
    for j, feed in enumerate(make_feed_from_data_set(
            data_set, input_ids, input_weights, input_length, target_labels, target_length
    )):
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, *_ = results
            summary_writer.add_summary(summary, epoch * batch_count + j)
        else:
            err = results[0]

        res.append(float(err))
        batch_number = j + 1
        if batch_number % 350 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(res)


def get_sample_len(sample):
    length = 0
    for it in sample:
        if isinstance(it, list):
            length += len(it)
        else:
            length += 1
    return length


def group_text_and_code_by_batches_v2(text, code, batch_size, text_tokens_with_end, code_tokens_with_end):
    text_end_marker = text_tokens_with_end - 1
    code_end_marker = code_tokens_with_end - 1
    text_start_marker = text_tokens_with_end - 2
    code_start_marker = code_tokens_with_end - 2

    def allign_batch(batch, end_marker, start_marker):
        lens = [get_sample_len(b) for b in batch]
        max_len = max(lens)
        result = [
            [start_marker] + sample + [end_marker for _ in range(max_len - sample_len + 1)]
            for sample, sample_len in zip(batch, lens)
        ]
        result_lens = np.asarray([ln + 2 for ln in lens])
        return result, result_lens

    text_with_code = list(zip(text, code))

    size = len(text_with_code) // batch_size
    batches = []
    for j in range(size):
        ind = j * batch_size
        d = text_with_code[ind:ind + batch_size]
        txt = [t for t, _ in d]
        cd = [c for _, c in d]
        txt = allign_batch(txt, text_end_marker, text_start_marker)
        cd = allign_batch(cd, code_end_marker, code_start_marker)
        batches.append((txt, cd))
    return batches


def seq2seq_text2text(text, code, num_text_tokens, num_code_tokens):
    batch_size = 1
    num_text_tokens_with_end = num_text_tokens + 2
    num_code_tokens_with_end = num_code_tokens + 2

    train_model = DummySeq2Seq.build_model(
        batch_size,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
        is_in_train_mode=True,
    )

    batches = group_text_and_code_by_batches_v2(
        text, code, batch_size, num_text_tokens_with_end,
        num_code_tokens_with_end
    )
    valid, train = split_batches(batches, 10)

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    train_summaries = build_loss_summary(train_model)
    train_updates = build_updates(train_model)

    initializer = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/gpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')

                tr_loss = run_seq2seq_text_model(data_set=train, model=train_model, session=sess, epoch=train_epoch,
                                                 is_train=True, updates=train_updates, summary_writer=summary_writer,
                                                 summaries=train_summaries)
                print('valid epoch')
                v_loss = run_seq2seq_text_model(data_set=valid, model=train_model, session=sess, epoch=train_epoch,
                                                is_train=False)
                saver.save(sess, 'models/model', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


SEQ_START_MARKER = 'SeqStartMarker'
SEQ_END_MARKER = 'SeqEndMarker'


def sort_via_sequences_length(data_set):
    data_set_with_length = [
        (ds, (get_sample_len(ds[0]), len(ds[1])))
        for ds in data_set
    ]

    def compare_fn(x_item, y_item):
        _, (x_size_first, x_size_second) = x_item
        _, (y_size_first, y_size_second) = y_item
        first_delta = abs(x_size_first - y_size_first)
        second_delta = abs(x_size_second - y_size_second)

        

    data_set_with_length.sort(key=functools.cmp_to_key(compare_fn))


def process_on_character_level(name):
    parsed = pd.read_csv(name)
    parsed = parsed[pd.notnull(parsed['question_text'])]
    parsed = parsed[pd.notnull(parsed['answer_text'])]
    parsed = parsed[pd.notnull(parsed['answer_code'])]

    inputs = [
        question + code + ' ' + answer
        for question, code, answer in zip(parsed['question_text'], parsed['question_code'], parsed['answer_text'])
    ]
    code = [it for it in parsed['answer_code']]

    code_character = chr(148)
    code_regexp = r'(<code>|</code>)'
    inputs = [re.sub(code_regexp, code_character, it) for it in inputs]
    inputs = [re.sub('\s+', ' ', it).strip() for it in inputs]

    characters = [ch for text in inputs + code for ch in list(text)]
    characters = count_items(characters, sort=True)
    characters = {ch for ch, count in characters if count > 20}

    data_set = [(c, i) for c, i in zip(code, inputs) if not set(c) - characters and not set(i) - characters]

    text_inputs = [text for _, text in data_set]
    code_inputs = [code for code, _ in data_set]

    words = [w for it in text_inputs for w in it.split()]
    words = count_items(words, sort=False)

    common_words = {w for w, cnt in words if cnt > 50}

    characters = list(characters)
    common_words_lst = list(common_words)
    word_mapping = {ch_or_word: i for i, ch_or_word in enumerate(sorted(characters + common_words_lst))}

    code_words = [w for it in code_inputs for w in it.split()]
    code_words = count_items(code_words, sort=True)

    code_common_words = {w for w, cnt in code_words if cnt > 50}
    unknown_token_marker = '___UNKNOWN__TOKEN___'
    code_mapping = {it: i for i, it in enumerate(sorted(code_common_words | {unknown_token_marker}))}
    unknown_token_id = code_mapping[unknown_token_marker]

    def text_as_word_indices(txt):
        splitted = txt.split()
        result = []
        for word in splitted:
            word_id = word_mapping.get(word, None)
            if word_id is not None:
                result.append(word_id)
                continue
            result.append([word_mapping[ch] for ch in word])
        return result

    indexed_text_inputs = [
        text_as_word_indices(text_input)
        for text_input in text_inputs
    ]

    indexed_code_inputs = [
        [
            code_mapping.get(cd, unknown_token_id)
            for cd in code_input
        ]
        for code_input in code_inputs
    ]

    data_set = list(zip(indexed_text_inputs, indexed_code_inputs))
    data_set = [d for d in data_set if get_sample_len(d[0]) < 200 and len(d[1]) < 100]

    indexed_text_inputs = [text for text, _ in data_set]
    indexed_code_inputs = [code for _, code in data_set]

    num_text_tokens = len(characters) + len(common_words_lst)
    num_code_tokens = len(code_common_words) + 1

    seq2seq_text2text(indexed_text_inputs, indexed_code_inputs, num_text_tokens, num_code_tokens)


if __name__ == '__main__':
    # parse('QueryResults.csv')
    # process_parsed('ParsedData.csv')
    process_on_character_level('ParsedData.csv')
