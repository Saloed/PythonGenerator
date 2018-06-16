import itertools
import json
import random
import re

import numpy as np
import tensorflow as tf

import DummySeq2Seq

from traceback import print_tb

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from analyze_django_prepare import SEQUENCE_END_TOKEN, WORD_PLACEHOLDER_TOKEN
from code_from_ast import generate_source
from utilss import dump_object, load_object, fix_r_index_keys, insert_words_into_placeholders

from net_conf import *
# from new_res.model_1_conf import *

# from results.model_9_net_conf import *

DATA_SET_FIELDS = [
    'id',
    'input',
    'copy_words',
    'code_target',
    'word_target',
]

BATCH_SIZE = 6


def align_batch(batch, seq_end_marker, time_major=True):
    lens = [len(b) for b in batch]
    max_len = max(lens)
    result = np.asarray([
        sample + [seq_end_marker for _ in range(max_len - sample_len)]
        for sample, sample_len in zip(batch, lens)
    ])
    if time_major:
        result = result.transpose([1, 0])
    return result, np.asarray(lens)


def destruct_data_set(data_set):
    result = []
    for i, name in enumerate(DATA_SET_FIELDS):
        result.append([it[i] for it in data_set])
    return tuple(result)


def construct_data_set(**kwargs):
    if not kwargs.keys() & set(DATA_SET_FIELDS):
        raise Exception('Incorrect data set')

    data_set = list(zip(*[kwargs[name] for name in DATA_SET_FIELDS]))
    return data_set


def make_copy_words(input, copy_word_mapping):
    input, _ = input
    return np.asarray([[copy_word_mapping.get(word, -1) for word in smt] for smt in input])


def preprocess_batch_fn_builder(
        input_end_marker, code_tar_end_marker, word_tar_end_marker, time_major=True
):
    def preprocess_batch(batch):
        _id, input, copy_words, code_tar, word_tar = destruct_data_set(batch)
        input = align_batch(input, input_end_marker, time_major)
        code_tar = align_batch(code_tar, code_tar_end_marker, time_major)
        word_tar = align_batch(word_tar, word_tar_end_marker, time_major)
        copy_words = align_batch(copy_words, -1, time_major)
        return _id, input, copy_words, code_tar, word_tar

    return preprocess_batch


def group_by_batches(data_set, batch_size, batch_preprocess_fn, shuffle=True):
    batch_count = len(data_set) // batch_size
    batches = []
    data_set.sort(key=lambda x: len(x[3]))
    for j in range(batch_count):
        ind = j * batch_size
        d = data_set[ind:ind + batch_size]
        processed = batch_preprocess_fn(d)
        batches.append(processed)
    if shuffle:
        random.shuffle(batches)
    return batches


def make_feed_from_data_set(data_set, model):
    for _id, (inputs, inputs_length), (copy_words, _), (code_tar, code_tar_len), (word_tar, word_tar_len) in data_set:
        yield _id, {
            model.inputs: inputs,
            model.input_len: inputs_length,
            model.code_target: code_tar,
            model.code_target_len: code_tar_len,
            model.word_target: word_tar,
            model.word_target_len: word_tar_len,
            model.copyable_input_ids: copy_words,
        }


def run_seq2seq_model(
        data_set, model, session, is_train, epoch=None,
        updates=None, summary_writer=None, summaries=None,
        need_outputs=False,
):
    fetches = [model.loss]
    if is_train:
        fetches += [summaries, updates]
    if need_outputs:
        fetches += [model.code_outputs, model.word_outputs, model.code_outputs_prob, model.word_outputs_prob]
    result_loss = []
    result_outputs = []
    batch_count = len(data_set)
    for j, (ids, feed) in enumerate(make_feed_from_data_set(data_set, model)):
        feed[model.enable_dropout] = is_train
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, *_ = results
            summary_writer.add_summary(summary, epoch * batch_count + j)
        else:
            err = results[0]

        if need_outputs:
            code_output, word_output, code_outputs_prob, word_outputs_prob = results[-4:]
            result_outputs.append((ids, code_output, word_output, code_outputs_prob, word_outputs_prob))

        result_loss.append(float(err))
        batch_number = j + 1
        if batch_number % 200 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(result_loss) if not need_outputs else (np.mean(result_loss), result_outputs)


def replace_time_major(batch, additional_axes=None):
    transpose_axes = [1, 0]
    if additional_axes:
        transpose_axes += additional_axes
    return np.asarray(batch).transpose(transpose_axes)


def replace_indexes_by_tokens(data, r_index):
    return [r_index[d] for d in data]


def ungroup_outputs(outputs,
                    targets=None, inputs=None, inputs_is_data_set=True,
                    targets_is_data_set=True, time_major=True):
    def ungroup_code_and_words(smth, additional_axes=None):
        if time_major:
            smth = (
                (replace_time_major(code, additional_axes), replace_time_major(words, additional_axes))
                for code, words in smth
            )
        samples = (sample for batch in smth for sample in zip(*batch))
        samples = [
            (code, words)
            for code, words in samples
        ]
        return samples

    def ungroup_inputs(smth):
        if time_major:
            smth = (replace_time_major(inp) for inp in smth)
        samples = (sample for batch in smth for sample in batch)
        samples = [
            desc
            for desc in samples
        ]
        return samples
    ids = [_id for _iid, *_ in outputs for _id in _iid]
    outputs_prob = [(code_prob, word_prob) for *_, code_prob, word_prob in outputs]
    outputs = [(code, word) for _, code, word, *_ in outputs]

    result_prob = ungroup_code_and_words(outputs_prob, [2])
    result = ungroup_code_and_words(outputs)
    result = ids, result, result_prob

    if targets is not None:
        if targets_is_data_set:
            target_lens = [
                (code_len, word_len)
                for *_, (_, code_tar_lens), (_, word_tar_lens) in targets
                for code_len, word_len in zip(code_tar_lens, word_tar_lens)
            ]
            targets = [(code_tar, word_tar) for *_, (code_tar, _), (word_tar, _) in targets]
        else:
            target_lens = None
        result = result + (ungroup_code_and_words(targets), target_lens)
    if inputs is not None:
        if inputs_is_data_set:
            inputs_len = [
                input_length
                for _, (inputs, inputs_length), *_ in inputs
                for input_length in inputs_length
            ]
            inputs = [inp for _, (inp, inputs_length), *_ in inputs]
        else:
            inputs_len = None

        result = result + (ungroup_inputs(inputs), inputs_len)
    return result


def eval_model(
        data_set,
        num_input_tokens,
        num_code_output_tokens,
        num_word_output_tokens,
        input_end_marker,
        code_output_end_marker,
        word_output_end_marker,
):
    batch_preprocess_fn = preprocess_batch_fn_builder(
        input_end_marker, code_output_end_marker, word_output_end_marker, time_major=True
    )
    batches = group_by_batches(data_set, BATCH_SIZE, batch_preprocess_fn, shuffle=False)
    model = DummySeq2Seq.build_model(
        BATCH_SIZE,
        num_input_tokens,
        num_code_output_tokens,
        num_word_output_tokens,
    )

    config = tf.ConfigProto()

    initializer = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        sess.run(initializer)
        saver.restore(sess, 'new_res/model_5')
        # saver.restore(sess, 'results/model_9_bleu_73')
        loss, outputs = run_seq2seq_model(
            data_set=batches,
            model=model,
            session=sess,
            is_train=False,
            need_outputs=True,
        )
    print(f'loss {loss}')
    result = ungroup_outputs(outputs, targets=batches, inputs=batches)

    dump_object(result, 'new_res/model_5_res')


def get_sequence_up_to_end(sequence, with_end=True):
    return list(itertools.takewhile(lambda x: x != SEQUENCE_END_TOKEN, sequence)) + (
        [SEQUENCE_END_TOKEN] if with_end else [])


def get_sources(code, words, is_generated):
    code = get_sequence_up_to_end(code)
    words = get_sequence_up_to_end(words, with_end=False)
    sources = generate_source([code], [words], is_generated)
    return sources[0]


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def compare_outputs_and_targets(outputs, target, target_len):
    code_len, words_len = target_len
    code_outputs, word_outputs = outputs
    code_targets, word_targets = target
    real_code_outputs, real_code_targets = code_outputs[:code_len], code_targets[:code_len]
    real_word_outputs, real_word_targets = word_outputs[:words_len], word_targets[:words_len]
    # combined = insert_words_into_placeholders(code_outputs, word_outputs, WORD_PLACEHOLDER_TOKEN)
    # combined_targets = insert_words_into_placeholders(code_targets, word_targets, WORD_PLACEHOLDER_TOKEN)
    # real_combined = insert_words_into_placeholders(real_code_outputs, real_word_outputs, WORD_PLACEHOLDER_TOKEN)
    # real_combined_targets = insert_words_into_placeholders(real_code_targets, real_word_targets, WORD_PLACEHOLDER_TOKEN)
    #
    # target_sources = get_sources(code_targets, word_targets, False)
    # generated_sources = get_sources(code_outputs, word_outputs, True)
    #
    # target_tokens = [tk for tk in target_sources.split()]
    # generated_tokens = [tk for tk in generated_sources.split()]
    # print('------------')
    # print(target_tokens)
    # print(generated_tokens)
    # target_tokens_bleu = tokenize_for_bleu_eval(target_sources)
    # generated_tokens_bleu = tokenize_for_bleu_eval(generated_sources)
    #
    # sm = SmoothingFunction()
    #
    # ngram_weights = [0.25] * min(4, len(target_tokens_bleu))
    # code_bleu = sentence_bleu([target_tokens_bleu], generated_tokens_bleu, weights=ngram_weights,
    #                           smoothing_function=sm.method3)

    return {
        'all_code_seq': sum(o == t for o, t in itertools.zip_longest(code_outputs, code_targets)),
        'code_seq_len': len(code_targets),
        'real_code_seq': sum(o == t for o, t in itertools.zip_longest(real_code_outputs, real_code_targets)),
        'code_real_len': code_len,
        'all_word_seq': sum(o == t for o, t in itertools.zip_longest(word_outputs, word_targets)),
        'word_seq_len': len(word_targets),
        'real_word_seq': sum(o == t for o, t in itertools.zip_longest(real_word_outputs, real_word_targets)),
        'word_real_len': words_len,
        # 'code': sum(o == t for o, t in zip(combined, combined_targets)),
        # 'code_len': len(combined_targets),
        # 'real_code': sum(o == t for o, t in zip(real_combined, real_combined_targets)),
        # 'read_code_len': code_len,
        # 'code_accuracy': sum(o == t for o, t in itertools.zip_longest(target_tokens, generated_tokens)),
        # 'code_accuracy_len': max(len(target_tokens), len(generated_tokens)),
        # 'code_bleu': code_bleu,
    }


def lookup_eval_result():
    result = load_object('new_res/model_5_res')
    ids, outputs, probs, targets, target_length, inputs, input_length = result
    stats = [
        compare_outputs_and_targets(out, tar, tar_len)
        for out, tar, tar_len in zip(outputs, targets, target_length)
    ]

    all_code_seq_percents = [st['all_code_seq'] / st['code_seq_len'] for st in stats]
    real_code_seq_percents = [st['real_code_seq'] / st['code_real_len'] for st in stats]

    all_word_seq_percents = [st['all_word_seq'] / st['word_seq_len'] for st in stats]
    real_word_seq_percents = [st['real_word_seq'] / st['word_real_len'] for st in stats]

    # code_percents = [st['code'] / st['code_len'] for st in stats]
    # real_code_percents = [st['real_code'] / st['read_code_len'] for st in stats]
    #
    # code_acc = [st['code_accuracy'] / st['code_accuracy_len'] for st in stats]
    # code_bleu = [st['code_bleu'] for st in stats]
    # real_code_bleu = [st['real_code_bleu'] for st in stats]

    print(f'code: all {np.mean(all_code_seq_percents)} real {np.mean(real_code_seq_percents)}')
    print(f'words: all {np.mean(all_word_seq_percents)} real {np.mean(real_word_seq_percents)}')
    # print(f'combined: all {np.mean(code_percents)} real {np.mean(real_code_percents)}')
    # print(f'bleu: all {np.mean(code_bleu)}')
    # print(f'accuracy: all {np.mean(code_acc)}')


def build_loss_summary(model):
    tf.summary.scalar('loss', model.loss)
    return tf.summary.merge_all()


def build_updates(model):
    loss = model.loss_with_l2
    # return tf.train.RMSPropOptimizer(0.003).minimize(loss)
    return tf.train.AdamOptimizer(0.0005).minimize(loss)


def train_model(
        train_data_set,
        valid_data_set,
        num_input_tokens,
        num_code_output_tokens,
        num_word_output_tokens,
        input_end_marker,
        code_output_end_marker,
        word_output_end_marker,
):
    batch_preprocess_fn = preprocess_batch_fn_builder(
        input_end_marker, code_output_end_marker, word_output_end_marker, time_major=True
    )
    train_batches = group_by_batches(train_data_set, BATCH_SIZE, batch_preprocess_fn)
    valid_batches = group_by_batches(valid_data_set, BATCH_SIZE, batch_preprocess_fn)

    model = DummySeq2Seq.build_model(
        BATCH_SIZE,
        num_input_tokens,
        num_code_output_tokens,
        num_word_output_tokens,
    )
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    train_summaries = build_loss_summary(model)
    train_updates = build_updates(model)

    initializer = tf.global_variables_initializer()

    encoder_variables_restorer = tf.train.Saver(var_list=encoder_variables)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        encoder_variables_restorer.restore(sess, 'pretrained/pretrain_{}_{}_{}'.format(
            ENCODER_INPUT_SIZE, ENCODER_STATE_SIZE, ENCODER_LAYERS))
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')

                tr_loss = run_seq2seq_model(
                    data_set=train_batches,
                    model=model,
                    session=sess,
                    epoch=train_epoch,
                    is_train=True,
                    updates=train_updates,
                    summary_writer=summary_writer,
                    summaries=train_summaries
                )
                print('valid epoch')
                v_loss = run_seq2seq_model(
                    data_set=valid_batches,
                    model=model,
                    session=sess,
                    epoch=train_epoch,
                    is_train=False
                )
                saver.save(sess, 'models/model', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def construct_data_sets(data_set):
    return {
        set_name: construct_data_set(
            id=data_set[set_name]['ids'],
            input=data_set[set_name]['descriptions'],
            code_target=data_set[set_name]['rules'],
            word_target=data_set[set_name]['words'],
            copy_words=data_set[set_name]['copy_words'],
        )
        for set_name in ('test', 'valid', 'train')
    }


def analyze(train):
    with open('django_data_set_x') as f:
        data_set = json.load(f)

    constructed_data_set = construct_data_sets(data_set)
    train_set = constructed_data_set['train']
    valid_set = constructed_data_set['valid']
    test_set = constructed_data_set['test']

    num_rule_tokens = data_set['train']['rules_size']
    num_description_tokens = data_set['train']['desc_size']
    num_word_tokens = data_set['train']['words_size']

    input_end_marker = data_set['train']['desc_seq_end']
    code_end_marker = data_set['train']['rules_seq_end']
    word_end_marker = data_set['train']['words_seq_end']

    print(
        len(train_set), len(valid_set), len(test_set),
        num_rule_tokens, num_description_tokens, num_word_tokens
    )

    if train:
        train_model(
            train_data_set=train_set,
            valid_data_set=valid_set,
            num_input_tokens=num_description_tokens,
            input_end_marker=input_end_marker,
            num_code_output_tokens=num_rule_tokens,
            code_output_end_marker=code_end_marker,
            num_word_output_tokens=num_word_tokens,
            word_output_end_marker=word_end_marker,
        )
    else:
        eval_model(
            data_set=test_set,
            num_input_tokens=num_description_tokens,
            input_end_marker=input_end_marker,
            num_code_output_tokens=num_rule_tokens,
            code_output_end_marker=code_end_marker,
            num_word_output_tokens=num_word_tokens,
            word_output_end_marker=word_end_marker,
        )


if __name__ == '__main__':
    # analyze(True)
    # analyze(False)
    lookup_eval_result()
