import json
from traceback import print_tb

import numpy as np
import tensorflow as tf

import DummySeq2Seq

from analyze_django_prepare import SEQUENCE_END_TOKEN
from utilss import dump_object, load_object

DATA_SET_FIELDS = [
    'input',
    'input_weight',
    'code_target',
    'word_target',
]

BATCH_SIZE = 6


def split_data_set(split_size, data_set):
    split_position = len(data_set) // split_size
    return data_set[:split_position], data_set[split_position:]


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

    return list(zip(*[kwargs[name] for name in DATA_SET_FIELDS]))


def preprocess_batch_fn_builder(
        input_end_marker, input_end_weight, code_tar_end_marker, word_tar_end_marker, time_major=True
):
    def preprocess_batch(batch):
        input, input_weight, code_tar, word_tar = destruct_data_set(batch)
        input = align_batch(input, input_end_marker, time_major)
        input_weight = align_batch(input_weight, input_end_weight, time_major)
        code_tar = align_batch(code_tar, code_tar_end_marker, time_major)
        word_tar = align_batch(word_tar, word_tar_end_marker, time_major)
        return input, input_weight, code_tar, word_tar

    return preprocess_batch


def group_by_batches(data_set, batch_size, batch_preprocess_fn):
    batch_count = len(data_set) // batch_size
    batches = []
    for j in range(batch_count):
        ind = j * batch_size
        d = data_set[ind:ind + batch_size]
        processed = batch_preprocess_fn(d)
        batches.append(processed)
    return batches


def make_feed_from_data_set(data_set, model):
    for (inputs, inputs_length), (input_weight, _), (code_tar, code_tar_len), (word_tar, word_tar_len) in data_set:
        yield {
            model.inputs: inputs,
            model.input_len: inputs_length,
            model.input_weight: input_weight,
            model.code_target: code_tar,
            model.code_target_len: code_tar_len,
            model.word_target: word_tar,
            model.word_target_len: word_tar_len,
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
        fetches += [model.code_outputs, model.word_outputs]
    result_loss = []
    result_outputs = []
    batch_count = len(data_set)
    for j, feed in enumerate(make_feed_from_data_set(data_set, model)):
        results = session.run(fetches=fetches, feed_dict=feed)
        if is_train:
            err, summary, *_ = results
            summary_writer.add_summary(summary, epoch * batch_count + j)
        else:
            err = results[0]

        if need_outputs:
            code_output, word_output = results[-2:]
            result_outputs.append((code_output, word_output))

        result_loss.append(float(err))
        batch_number = j + 1
        if batch_number % 200 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(result_loss) if not need_outputs else (np.mean(result_loss), result_outputs)


def replace_time_major(batch):
    return np.asarray(batch).transpose([1, 0])


def ungroup_outputs(outputs, targets=None, targets_is_data_set=True, time_major=True):
    def ungroup(smth):
        if time_major:
            smth = [(replace_time_major(code), replace_time_major(words)) for code, words in smth]
        return [sample for batch in smth for sample in zip(*batch)]

    result = ungroup(outputs)
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
        result = result, ungroup(targets), target_lens
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
        input_end_marker, 1.0, code_output_end_marker, word_output_end_marker, time_major=True
    )
    batches = group_by_batches(data_set, BATCH_SIZE, batch_preprocess_fn)
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
        saver.restore(sess, 'models/model-6')
        loss, outputs = run_seq2seq_model(
            data_set=batches,
            model=model,
            session=sess,
            is_train=False,
            need_outputs=True,
        )
    print(f'loss {loss}')
    result = ungroup_outputs(outputs, targets=batches)
    dump_object(result, 'model_new_res')


def compare_outputs_and_targets(outputs, target, target_len):
    code_len, words_len = target_len
    code_outputs, word_outputs = outputs
    code_targets, word_targets = target
    real_code_outputs, real_code_targets = code_outputs[:code_len], code_targets[:code_len]
    real_word_outputs, real_word_targets = word_outputs[:words_len], word_targets[:words_len]

    return {
        'all_code_seq': sum(o == t for o, t in zip(code_outputs, code_targets)),
        'code_seq_len': len(code_targets),
        'real_code_seq': sum(o == t for o, t in zip(real_code_outputs, real_code_targets)),
        'code_real_len': code_len,
        'all_word_seq': sum(o == t for o, t in zip(word_outputs, word_targets)),
        'word_seq_len': len(word_targets),
        'real_word_seq': sum(o == t for o, t in zip(real_word_outputs, real_word_targets)),
        'word_real_len': words_len,
    }


def lookup_eval_result():
    result = load_object('model_new_res')
    outputs, targets, target_length = result
    stats = [
        compare_outputs_and_targets(out, tar, tar_len) for out, tar, tar_len in zip(outputs, targets, target_length)
    ]

    all_code_seq_percents = [st['all_code_seq'] / st['code_seq_len'] for st in stats]
    real_code_seq_percents = [st['real_code_seq'] / st['code_real_len'] for st in stats]

    all_word_seq_percents = [st['all_word_seq'] / st['word_seq_len'] for st in stats]
    real_word_seq_percents = [st['real_word_seq'] / st['word_real_len'] for st in stats]

    print(f'code: all {np.mean(all_code_seq_percents)} real {np.mean(real_code_seq_percents)}')
    print(f'words: all {np.mean(all_word_seq_percents)} real {np.mean(real_word_seq_percents)}')


def build_loss_summary(model):
    tf.summary.scalar('loss', model.loss)
    return tf.summary.merge_all()


def build_updates(model):
    loss = model.loss
    # return tf.train.RMSPropOptimizer(0.003).minimize(loss)
    return tf.train.AdamOptimizer().minimize(loss)


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
        input_end_marker, 1.0, code_output_end_marker, word_output_end_marker, time_major=True
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
        encoder_variables_restorer.restore(sess, 'models/pretrain-4')
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


def filter_big_sequences(data_set):
    return [d for d in data_set if len(d[0]) < 200]


def analyze(train):
    with open('django_data_set_3.json') as f:
        data_set = json.load(f)

    constructed_data_set = construct_data_set(
        input=data_set['indexed_description'],
        input_weight=data_set['description_weights'],
        code_target=data_set['indexed_ast'],
        word_target=data_set['indexed_words'],
    )

    constructed_data_set = filter_big_sequences(constructed_data_set)

    valid_set, train_set = split_data_set(10, constructed_data_set)

    num_ast_tokens = len(data_set['ast_token_index'])
    num_description_tokens = len(data_set['desc_word_index'])
    num_word_tokens = len(data_set['words_index'])

    print(len(train_set), len(valid_set), num_ast_tokens, num_description_tokens, num_word_tokens)
    if train:
        train_model(
            train_data_set=train_set,
            valid_data_set=valid_set,
            num_input_tokens=num_description_tokens,
            input_end_marker=data_set['desc_word_index'][SEQUENCE_END_TOKEN],
            num_code_output_tokens=num_ast_tokens,
            code_output_end_marker=data_set['ast_token_index'][SEQUENCE_END_TOKEN],
            num_word_output_tokens=num_word_tokens,
            word_output_end_marker=data_set['words_index'][SEQUENCE_END_TOKEN],
        )
    else:
        eval_model(
            data_set=valid_set,
            num_input_tokens=num_description_tokens,
            input_end_marker=data_set['desc_word_index'][SEQUENCE_END_TOKEN],
            num_code_output_tokens=num_ast_tokens,
            code_output_end_marker=data_set['ast_token_index'][SEQUENCE_END_TOKEN],
            num_word_output_tokens=num_word_tokens,
            word_output_end_marker=data_set['words_index'][SEQUENCE_END_TOKEN],
        )


if __name__ == '__main__':
    # analyze(True)
    # analyze(False)
    lookup_eval_result()
