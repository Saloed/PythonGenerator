import sys
import ast
import html
import random
import re
import _pickle as P

import timeout_decorator as td
import pandas as pd
import numpy as np
import tensorflow as tf

import DummySeq2Seq
import spizjenno_runner
import tree_transformers

from collections import defaultdict
from traceback import print_tb
from multiprocessing import Pool

from tree_transformers import CONSTANT_LITERAL_TYPE
from some_net import convert_to_node


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

#
# import matplotlib.pyplot as plotter
#
#
# def new_figure(num, epoch_in_retry, max_y):
#     x = np.arange(0, epoch_in_retry, 1)
#     yv = np.full(epoch_in_retry, -1.1)
#     yt = np.full(epoch_in_retry, -1.1)
#     fig = plotter.figure(num)
#     fig.set_size_inches(10, 10)
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_xlim(0, epoch_in_retry)
#     ax.set_ylim(0, max_y)
#     ax.set_xlabel('epoch')
#     ax.set_ylabel('error')
#     ax.grid(True)
#     line = ax.plot(x, yv, 'r.', x, yt, 'b.')
#     fig.show(False)
#     fig.canvas.draw()
#     return line, fig
#
#
# def update_figure(plot, axes, x, yv, yt):
#     new_data = axes[0].get_ydata()
#     new_data[x] = yv
#     axes[0].set_ydata(new_data)
#     new_data = axes[1].get_ydata()
#     new_data[x] = yt
#     axes[1].set_ydata(new_data)
#     plot.canvas.draw()


def shuffle_data_set(code, text):
    combined = [(c, t) for c, t in zip(code, text)]
    random.shuffle(combined)
    shuffled_code = [c for c, _ in combined]
    shuffled_text = [t for _, t in combined]
    return shuffled_code, shuffled_text


def rand_int_except(i, bot, top):
    res = random.randint(bot, top)
    while res == i:
        res = random.randint(bot, top)
    return res


def make_samples(text, positive, negative, samples_cnt):
    for i, (txt, code) in enumerate(zip(text, positive)):
        for _ in range(samples_cnt):
            neg = negative[rand_int_except(i, 0, len(negative) - 1)]
            yield txt, code, neg


def make_samples_feed(text, positive, negative, samples_cnt):
    for txt, pos, neg in make_samples(text, positive, negative, samples_cnt):
        yield {k: v for k, v in list(pos.items()) + list(txt.items()) + list(neg.items())}


def split_batches(batches, split_size):
    split_position = len(batches) // split_size
    return batches[:split_position], batches[split_position:]


def get_sample_embs(samples, num_tokens):
    return np.asarray(samples), np.asarray([
        [
            np.eye(num_tokens)[it]
            for it in sample
        ]
        for sample in samples
    ])


def group_text_by_batches(text, batch_size, num_tokens, time_major=False):
    end_marker = num_tokens - 1

    def allign_batch(batch):
        lens = [len(b) for b in batch]
        max_len = max(lens)
        result_samples = [
            sample + [end_marker for _ in range(max_len - sample_len + 1)]
            for sample, sample_len in zip(batch, lens)
        ]
        result = result_samples
        if time_major:
            result = [[] for _ in range(max_len + 1)]
            for j in range(max_len + 1):
                for sample in result_samples:
                    result[j].append(sample[j])

        result_lens = np.asarray([ln + 1 for ln in lens])
        return result, result_lens

    size = len(text) // batch_size
    batches = []
    for j in range(size):
        ind = j * batch_size
        d = text[ind:ind + batch_size]
        d = allign_batch(d)
        batches.append(d)
    return batches


def group_text_and_code_by_batches(text, code, batch_size, text_tokens_with_end, code_tokens_with_end):
    text_end_marker = text_tokens_with_end - 1
    code_end_marker = code_tokens_with_end - 1
    text_start_marker = text_tokens_with_end - 2
    code_start_marker = code_tokens_with_end - 2

    def allign_batch(batch, end_marker, start_marker):
        lens = [len(b) for b in batch]
        max_len = max(lens)
        result_samples = [
            [start_marker] + sample + [end_marker for _ in range(max_len - sample_len + 1)]
            for sample, sample_len in zip(batch, lens)
        ]
        result = [[] for _ in range(max_len + 2)]
        for j in range(max_len + 2):
            for sample in result_samples:
                result[j].append(sample[j])

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


def run_seq2seq_model(data_set, model, session, epoch, is_train, updates=None, summary_writer=None, summaries=None):
    inputs, input_length, target_labels, target_length, outputs, loss = model
    fetches = [loss]
    if is_train:
        fetches += [summaries, updates]
    res = []
    batch_count = len(data_set)
    for j, ((text, text_lens), (code, code_lens)) in enumerate(data_set):
        feed = {
            inputs: text,
            input_length: text_lens,
            target_labels: code,
            target_length: code_lens,
        }
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, _ = results
            summary_writer.add_summary(summary, epoch * batch_count + j)
        else:
            err = results[0]

        res.append(float(err))
        batch_number = j + 1
        if batch_number % 350 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(res)


def build_loss_summary(model):
    *_, loss = model
    tf.summary.scalar('loss', loss)
    return tf.summary.merge_all()


def build_updates(model):
    *_, loss = model
    # return tf.train.RMSPropOptimizer(0.005).minimize(loss)
    return tf.train.AdamOptimizer().minimize(loss)


def run_with_outputs(
        data_set,
        model,
        session,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
):
    inputs, input_length, target_labels, target_length, outputs, loss = model
    res = []
    for ((text, text_lens), (code, code_lens)) in data_set:
        # text, text_embs = get_sample_embs(text, num_text_tokens_with_end)
        # code, code_embs = get_sample_embs(code, num_code_tokens_with_end)
        feed = {
            inputs: text,
            input_length: text_lens,
            target_labels: code,
            target_length: code_lens,
        }
        output = session.run(fetches=outputs, feed_dict=feed)
        res.append((output, code))
    return res


def fix_results(results):
    res = []

    def flat(smth):
        flatten = [[] for _ in range(len(smth[0]))]
        for batch_at_time in smth:
            for i, it in enumerate(batch_at_time):
                flatten[i].append(it)
        return flatten

    for outputs, targets in results:
        outputs = flat(outputs)
        targets = flat(targets)
        res += list(zip(outputs, targets))
    return res


def seq2seq_text2code_results(text, code, num_text_tokens, num_code_tokens):
    batch_size = 1
    num_text_tokens_with_end = num_text_tokens + 2
    num_code_tokens_with_end = num_code_tokens + 2

    # model = Seq2Seq.build_model(
    #     batch_size,
    #     num_text_tokens_with_end,
    #     num_code_tokens_with_end,
    #     is_in_train_mode=True,
    # )

    model = spizjenno_runner.run(num_text_tokens_with_end, num_code_tokens_with_end, batch_size)

    batches = group_text_and_code_by_batches(text, code, batch_size, num_text_tokens_with_end, num_code_tokens_with_end)
    valid, train = split_batches(batches, 10)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        saver.restore(sess, 'models/model-0')
        results = run_with_outputs(valid, model, sess, num_text_tokens_with_end, num_code_tokens_with_end)
        results = fix_results(results)
        with open('some_results.dmp', 'wb') as f:
            P.dump(results, f)


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


def seq2seq_text2code(text, code, num_text_tokens, num_code_tokens):
    batch_size = 1
    num_text_tokens_with_end = num_text_tokens + 2
    num_code_tokens_with_end = num_code_tokens + 2

    # train_model = Seq2Seq.build_model(
    #     batch_size,
    #     num_text_tokens_with_end,
    #     num_code_tokens_with_end,
    #     is_in_train_mode=True,
    # )
    # valid_model = Seq2Seq.build_model(
    #     batch_size,
    #     num_text_tokens_with_end,
    #     num_code_tokens_with_end,
    #     is_in_train_mode=False,
    # )

    # train_model = spizjenno_runner.run(num_text_tokens_with_end, num_code_tokens_with_end, batch_size)

    train_model = DummySeq2Seq.build_model(
        batch_size,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
        is_in_train_mode=True,
    )

    batches = group_text_and_code_by_batches(text, code, batch_size, num_text_tokens_with_end, num_code_tokens_with_end)
    valid, train = split_batches(batches, 10)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    train_summaries = build_loss_summary(train_model)
    train_updates = build_updates(train_model)

    initializer = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=config) as sess, tf.device('/gpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                tr_loss = run_seq2seq_model(data_set=train, model=train_model, session=sess, epoch=train_epoch,
                                            is_train=True, updates=train_updates, summary_writer=summary_writer,
                                            summaries=train_summaries)
                print('valid epoch')
                v_loss = run_seq2seq_model(data_set=valid, model=train_model, session=sess, epoch=train_epoch,
                                           is_train=False)
                saver.save(sess, 'models/model', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def get_word_embedding(word, emb_matrix):
    if isinstance(word, list):
        size = len(word)
        return [emb_matrix[ch] / size for ch in word]
    return [emb_matrix[word]]


def make_feed_from_data_set(data_set, embedding_size, inputs, input_length, target_labels, target_length):
    emb_matrix = np.eye(embedding_size)
    for (text, text_lens), (code, code_lens) in data_set:
        text_embs = [
            [
                emb
                for word in sample
                for emb in get_word_embedding(word, emb_matrix)
             ]
            for sample in text
        ]
        text_inputs = np.asarray(text_embs).transpose([1, 0, 2])
        code_inputs = np.asarray(code).transpose([1, 0])
        yield {
            inputs: text_inputs,
            input_length: text_lens,
            target_labels: code_inputs,
            target_length: code_lens,
        }


def run_seq2seq_text_model(data_set, model, session, epoch, is_train, num_text_tokens, updates=None, summary_writer=None, summaries=None):
    inputs, input_length, target_labels, target_length, outputs, loss = model
    fetches = [loss]
    if is_train:
        fetches += [summaries, updates]
    res = []
    batch_count = len(data_set)
    for j, feed in enumerate(
            make_feed_from_data_set(data_set, num_text_tokens, inputs, input_length, target_labels, target_length)
    ):
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, _ = results
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

    batches = group_text_and_code_by_batches_v2(text, code, batch_size, num_text_tokens_with_end, num_code_tokens_with_end)
    valid, train = split_batches(batches, 10)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    train_summaries = build_loss_summary(train_model)
    train_updates = build_updates(train_model)

    initializer = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=config) as sess, tf.device('/gpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                tr_loss = run_seq2seq_text_model(data_set=train, model=train_model, session=sess, epoch=train_epoch,
                                            is_train=True, updates=train_updates, summary_writer=summary_writer,
                                            summaries=train_summaries, num_text_tokens=num_text_tokens_with_end)
                print('valid epoch')
                v_loss = run_seq2seq_text_model(data_set=valid, model=train_model, session=sess, epoch=train_epoch,
                                           is_train=False, num_text_tokens=num_text_tokens_with_end)
                saver.save(sess, 'models/model', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def tokens_from_code(code):
    tokens = set()
    for _code in code:
        tokens |= {tk.token_type for tk in _code.all_nodes}
    return list(sorted(tokens))


SEQ_START_MARKER = 'SeqStartMarker'
SEQ_END_MARKER = 'SeqEndMarker'


def make_indexed_code(code, tokens):
    token_idx = {tk: i for i, tk in enumerate(tokens)}
    token_idx[SEQ_START_MARKER] = len(token_idx)
    token_idx[SEQ_END_MARKER] = len(token_idx)

    def process_tree(tree, result):
        result.append(tree.token_type)
        if tree.children:
            result.append(SEQ_START_MARKER)
            for ch in tree.children:
                process_tree(ch, result)
            result.append(SEQ_END_MARKER)
        return result

    code_tree_as_list = [process_tree(it.root_node, []) for it in code]
    indexed_code = [[token_idx[it] for it in cd] for cd in code_tree_as_list]
    r_index = {idx: tk for tk, idx in token_idx.items()}
    return indexed_code, r_index


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
    random.shuffle(data_set)

    indexed_text_inputs = [text for text, _ in data_set]
    indexed_code_inputs = [code for _, code in data_set]

    num_text_tokens = len(characters) + len(common_words_lst)
    num_code_tokens = len(code_common_words) + 1

    seq2seq_text2text(indexed_text_inputs, indexed_code_inputs, num_text_tokens, num_code_tokens)


def process_parsed(name):
    parsed = pd.read_csv(name)
    parsed = parsed[pd.notnull(parsed['question_text'])]
    parsed = parsed[pd.notnull(parsed['answer_text'])]
    parsed = parsed[pd.notnull(parsed['answer_code'])]
    inputs = [question + answer for question, answer in zip(parsed['question_text'], parsed['answer_text'])]
    code = [(ast.parse(it), it) for it in parsed['answer_code']]

    code_common_names = [extract_common_names(*tree) for tree in code]

    attribute_names, function_names = [], []
    for _, (attributes, functions) in code_common_names:
        attribute_names += attributes.values()
        function_names += functions.values()

    attribute_names = get_most_common_names(attribute_names)
    function_names = get_most_common_names(function_names)

    # renamed_code = [
    #     (replace_names(tree, attribute_names, function_names), text)
    #     for tree, text in code
    # ]
    renamed_code = code

    transformed_code = [
        (transform_tree(tree, text), text)
        for tree, text in renamed_code
    ]

    trees = [tree for tree, _ in transformed_code]

    transformed_code = [
        replace_empty_tokens(tree)
        for tree in trees
    ]

    transformed_code = [
        children_as_list(tree)
        for tree in transformed_code
    ]

    # non_terminals = [find_non_terminals(it) for it, _ in transformed_code]
    tokens = extract_tokens(transformed_code)
    code_as_nodes = [convert_to_node(tree) for tree in transformed_code]

    characters = [ch for text in inputs for ch in list(text)]
    characters = count_items(characters, sort=True)
    characters = {ch for ch, count in characters if count > 20}

    data_set = [(c, i) for c, i in zip(code_as_nodes, inputs) if c.non_leafs and not set(i) - characters]
    text_inputs = [text for _, text in data_set]
    code_inputs = [code for code, _ in data_set]

    characters = list(characters)
    character_mapping = {ch: i for i, ch in enumerate(sorted(characters))}

    indexed_text_inputs = [
        [
            character_mapping[ch]
            for ch in text_input
        ]
        for text_input in text_inputs
    ]

    num_text_tokens = len(characters)
    code_tokens = tokens_from_code(code_inputs)
    indexed_code_inputs, code_tokens_r_index = make_indexed_code(code_inputs, code_tokens)
    num_code_tokens = len(code_tokens_r_index)

    data_set = list(zip(indexed_text_inputs, indexed_code_inputs))
    # data_set.sort(key=lambda it: len(it[1]))
    data_set = [d for d in data_set if len(d[0]) < 200 and len(d[1]) < 100]
    random.shuffle(data_set)
    indexed_text_inputs = [text for text, _ in data_set]
    indexed_code_inputs = [code for _, code in data_set]

    # literals = [
    #     literal
    #     for tree, _ in transformed_code
    #     for literal in get_constant_literals(tree)
    # ]
    #
    # literals = count_items(literals, sort=True)
    # literals_bpe = smth_like_bpe.make_bpe(literals)

    # seq2seq_input_check(num_text_tokens, indexed_text_inputs)
    seq2seq_text2code(indexed_text_inputs, indexed_code_inputs, num_text_tokens, num_code_tokens)
    # seq2seq_text2code_results(indexed_text_inputs, indexed_code_inputs, num_text_tokens, num_code_tokens)
    # some_analysis(code_tokens_r_index)


if __name__ == '__main__':
    # parse('QueryResults.csv')
    # process_parsed('ParsedData.csv')
    process_on_character_level('ParsedData.csv')
