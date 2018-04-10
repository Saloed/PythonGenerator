import ast
import html
import random
import re
from collections import defaultdict
from traceback import print_tb

import timeout_decorator as td
import pandas as pd
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

import sys

import smth_like_bpe
import tree_transformers
from some_net_stuff import Encoder
from some_net_stuff.BatchBuilder import generate_batches
from some_net_stuff.NetBuilder import build_net
from some_net_stuff.TFParameters import init_params
from tree_transformers import CONSTANT_LITERAL_TYPE
from some_net import convert_to_node
from some_net_stuff import Seq2Seq


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


import matplotlib.pyplot as plotter


def new_figure(num, epoch_in_retry, max_y):
    x = np.arange(0, epoch_in_retry, 1)
    yv = np.full(epoch_in_retry, -1.1)
    yt = np.full(epoch_in_retry, -1.1)
    fig = plotter.figure(num)
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, epoch_in_retry)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.grid(True)
    line = ax.plot(x, yv, 'r.', x, yt, 'b.')
    fig.show(False)
    fig.canvas.draw()
    return line, fig


def update_figure(plot, axes, x, yv, yt):
    new_data = axes[0].get_ydata()
    new_data[x] = yv
    axes[0].set_ydata(new_data)
    new_data = axes[1].get_ydata()
    new_data[x] = yt
    axes[1].set_ydata(new_data)
    plot.canvas.draw()


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


def get_sample_embs(samples, num_tokens, just_zero=False):
    return np.asarray(samples), np.asarray([
        [
            np.eye(num_tokens)[it] if not just_zero else np.zeros([num_tokens])
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


def seq2seq_input_check(num_text_inputs, indexed_text_inputs):
    batch_size = 10
    num_text_inputs_with_end = num_text_inputs + 1
    inputs, input_length, targets, target_labels, target_length, loss = Seq2Seq.build_model(
        batch_size,
        num_text_inputs_with_end,
        num_text_inputs_with_end,
    )
    text_batches = group_text_by_batches(indexed_text_inputs, batch_size, num_text_inputs_with_end, True)
    valid, train = split_batches(text_batches, 10)
    updates = tf.train.AdamOptimizer().minimize(loss)
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    initializer = tf.global_variables_initializer()
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()

    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                res = []
                batch_count = len(train)
                for j, (sample, sample_lens) in enumerate(train):
                    sample, sample_embs = get_sample_embs(sample, num_text_inputs_with_end)
                    err, summary, _ = sess.run(fetches=[loss, summaries, updates], feed_dict={
                        inputs: sample_embs,
                        input_length: sample_lens,
                        targets: sample_embs,
                        target_labels: sample,
                        target_length: sample_lens,
                    })
                    summary_writer.add_summary(summary, train_epoch * batch_count + j)

                    res.append(float(err))
                    batch_number = j + 1
                    if batch_number % 100 == 0:
                        percent = int(j / batch_count * 100)
                        print(f'Complete {percent}')

                tr_loss = np.mean(res)
                print('valid epoch')
                res = []
                for sample, sample_lens in valid:
                    sample, sample_embs = get_sample_embs(sample, num_text_inputs_with_end)
                    err, *_ = sess.run(fetches=[loss], feed_dict={
                        inputs: sample_embs,
                        input_length: sample_lens,
                        targets: sample_embs,
                        target_labels: sample,
                        target_length: sample_lens,
                    })
                    res.append(float(err))

                v_loss = np.mean(res)

                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


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
        for j in range(max_len + 1):
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


def run_seq2seq_model(
        data_set,
        model,
        session,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
        epoch,
        is_train,
        updates=None,
        summary_writer=None,
        summaries=None
):
    inputs, input_length, targets, target_labels, target_length, loss = model
    fetches = [loss]
    if is_train:
        fetches += [summaries, updates]
    res = []
    batch_count = len(data_set)
    for j, ((text, text_lens), (code, code_lens)) in enumerate(data_set):
        text, text_embs = get_sample_embs(text, num_text_tokens_with_end)
        code, code_embs = get_sample_embs(code, num_code_tokens_with_end)
        feed = {
            inputs: text_embs,
            input_length: text_lens,
            target_labels: code,
            target_length: code_lens,
        }
        if is_train:
            feed[targets] = code_embs
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, _ = results
            summary_writer.add_summary(summary, epoch * batch_count + j)
        else:
            err = results[0]

        res.append(float(err))
        batch_number = j + 1
        if batch_number % 100 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(res)


def build_loss_summary(model):
    *_, loss = model
    tf.summary.scalar('loss', loss)
    return tf.summary.merge_all()


def build_updates(model):
    *_, loss = model
    return tf.train.AdamOptimizer().minimize(loss)


def seq2seq_text2code(text, code, num_text_tokens, num_code_tokens):
    batch_size = 10
    num_text_tokens_with_end = num_text_tokens + 2
    num_code_tokens_with_end = num_code_tokens + 2

    train_model = Seq2Seq.build_model(
        batch_size,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
        is_in_train_mode=True,
    )
    valid_model = Seq2Seq.build_model(
        batch_size,
        num_text_tokens_with_end,
        num_code_tokens_with_end,
        is_in_train_mode=False,
    )

    batches = group_text_and_code_by_batches(text, code, batch_size, num_text_tokens_with_end, num_code_tokens_with_end)
    valid, train = split_batches(batches, 10)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    initializer = tf.global_variables_initializer()
    train_summaries = build_loss_summary(train_model)
    train_updates = build_updates(train_model)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                tr_loss = run_seq2seq_model(
                    data_set=train,
                    model=train_model,
                    session=sess,
                    num_text_tokens_with_end=num_text_tokens_with_end,
                    num_code_tokens_with_end=num_code_tokens_with_end,
                    epoch=train_epoch,
                    is_train=True,
                    updates=train_updates,
                    summary_writer=summary_writer,
                    summaries=train_summaries,
                )
                print('valid epoch')
                v_loss = run_seq2seq_model(
                    data_set=valid,
                    model=valid_model,
                    session=sess,
                    num_text_tokens_with_end=num_text_tokens_with_end,
                    num_code_tokens_with_end=num_code_tokens_with_end,
                    epoch=train_epoch,
                    is_train=False,
                )
                saver.save(sess, 'models/model', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def pretrain_with_negative_samples(tokens, code_inputs, characters, indexed_text_inputs):
    params, emb_indexes = init_params(tokens)
    net = build_net(params)
    negative_net = build_net(params)
    code_batches = generate_batches(code_inputs, emb_indexes, net, 0.8)
    negative_code_batches = generate_batches(code_inputs, emb_indexes, negative_net, 1.0)

    text_input_pc, encoder_out, empty_token_idx = Encoder.get_net(len(characters))
    text_input_batches = Encoder.generate_batches(indexed_text_inputs, text_input_pc, empty_token_idx)

    distance_delta = tf.constant(5.0, tf.float32)

    # negative_net_out = tf.stop_gradient(negative_net.out)
    negative_net_out = negative_net.out

    positive_net_out = net.out
    positive_distance = tf.norm(encoder_out - positive_net_out, axis=1)

    # negative_distance = tf.stop_gradient(tf.norm(encoder_out - negative_net_out))
    negative_distance = tf.norm(encoder_out - negative_net_out, axis=1)

    # tf.summary.histogram('encoder_out', encoder_out)
    # tf.summary.histogram('positive_out', positive_net_out)
    # tf.summary.histogram('negative_out', negative_net_out)

    # loss = positive_distance / (negative_distance + 1e-6)

    loss = positive_distance + distance_delta - negative_distance
    loss = tf.nn.relu(loss)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('positive', tf.reduce_mean(positive_distance))
    tf.summary.scalar('negative', tf.reduce_mean(negative_distance))
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()
    weights = [
        var
        for var in tf.trainable_variables()
        if len(var.shape) > 1 and var.shape[0] > 1
           and not var.name.startswith('rnn')
    ]
    regularized_weights = [
        tf.nn.l2_loss(w)
        for w in weights
    ]
    l2_loss = tf.reduce_sum(regularized_weights)
    # updates = tf.train.AdamOptimizer().minimize(loss + 1e-3 * l2_loss)
    updates = tf.train.RMSPropOptimizer(
        learning_rate=0.001,
        decay=0.1,
        momentum=0.9,
        centered=True,
    ).minimize(loss + 1e-3 * l2_loss)

    validation_code, train_code = split_batches(code_batches, 10)
    valid_negative, train_negative = split_batches(negative_code_batches, 10)
    validation_text, train_text = split_batches(text_input_batches, 10)

    initializer = tf.global_variables_initializer()
    for retry_num in range(1):
        # plot_axes, plot = new_figure(retry_num, 100, 2)
        saver = tf.train.Saver(max_to_keep=10)
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter('models', sess.graph)
            sess.run(initializer)
            try:
                for train_epoch in range(100):
                    print(f'start epoch {train_epoch}')
                    res = []
                    for j, feed in enumerate(make_samples_feed(train_text, train_code, train_negative, 3)):
                        err, summary, _ = sess.run(fetches=[loss, summaries, updates], feed_dict=feed)
                        summary_writer.add_summary(summary, train_epoch * 3000 + j)
                        res.append(float(err))
                        # if (j + 1) % 50 == 0:
                        #     print(f'iteration {j}')

                    tr_loss = np.mean(res)
                    print('valid epoch')
                    res = []
                    for j, feed in enumerate(make_samples_feed(validation_text, validation_code, valid_negative, 1)):
                        err, *_ = sess.run(fetches=[loss], feed_dict=feed)
                        res.append(float(err))

                    v_loss = np.mean(res)

                    print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')
                    saver.save(sess, 'models/model', train_epoch)

                    # update_figure(plot, plot_axes, train_epoch, v_loss, tr_loss)
            except Exception as ex:
                print(ex)
                # print_tb(ex.__traceback__)


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
    data_set.sort(key=lambda x: len(x[1]))
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

    #
    # import ipdb
    # ipdb.set_trace()
    exit(99)


if __name__ == '__main__':
    # parse('QueryResults.csv')
    process_parsed('ParsedData.csv')
