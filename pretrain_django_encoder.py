import json
from traceback import print_tb

import numpy as np
import tensorflow as tf
from tensorflow import variable_scope as vs

from prepare_ast_to_tree import convert_trees_to_node
from analyze_django_prepare import SEQUENCE_END_TOKEN, SUBTREE_START_TOKEN
from some_net_stuff.BatchBuilder import prepare_sample
from DummySeq2Seq import build_encoder
from some_net_stuff.TFParameters import init_params
from some_net_stuff.NetBuilder import build_net

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


def make_batches(data_set, input_end_marker):
    batch_count = len(data_set) // BATCH_SIZE
    batches = []
    for j in range(batch_count):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        desc = [dsc for dsc, _ in d]
        code = [c for _, c in d]
        desc = align_batch(desc, input_end_marker)
        batches.append((desc, code))
    return batches


def make_feed_from_data_set(data_set, encoder_pc, code_pc):
    for (inputs, inputs_length), code in data_set:
        pc = {
            encoder_pc['inputs']: inputs,
            encoder_pc['input_len']: inputs_length,
        }
        for i in range(BATCH_SIZE):
            pc.update(code_pc[i].assign(code[i]))

        yield pc


def run_pretrain(
        data_set,
        loss,
        session,
        updates,
        encoder_pc,
        code_pc,
):
    result_loss = []
    batch_count = len(data_set)
    fetches = [loss, updates]
    for j, feed in enumerate(make_feed_from_data_set(data_set, encoder_pc, code_pc)):
        err, *_ = session.run(fetches=fetches, feed_dict=feed)
        result_loss.append(float(err))
        batch_number = j + 1
        if batch_number % 200 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(result_loss)


def train(data_set, description_num_tokens, code_num_tokens, input_end_marker):
    with vs('encoder') as e_scope:
        input_ids = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'inputs')
        input_length = tf.placeholder(tf.int32, [BATCH_SIZE], 'input_len')

        encoder_output, encoder_state_fw, encoder_state_bw = build_encoder(
            input_ids, input_length, description_num_tokens, 128, 64, 1, e_scope, 0.8
        )
        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]
        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)
    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

    code_analyzer_params = init_params(code_num_tokens)
    code_analyzer = build_net(code_analyzer_params)

    positive_net_out = code_analyzer.out
    negative_net_out = tf.random_shuffle(positive_net_out)
    negative_net_out = tf.stop_gradient(negative_net_out)
    positive_distance = tf.norm(encoder_result - positive_net_out, axis=1)
    negative_distance = tf.norm(encoder_result - negative_net_out, axis=1)
    loss = positive_distance + 1.0 - negative_distance
    loss = tf.nn.relu(loss)
    loss = tf.reduce_mean(loss)
    update = tf.train.AdamOptimizer().minimize(loss)

    batches = make_batches(data_set, input_end_marker)

    initializer = tf.global_variables_initializer()
    config = tf.ConfigProto()
    saver = tf.train.Saver(var_list=encoder_variables, max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')

                tr_loss = run_pretrain(
                    data_set=batches,
                    loss=loss,
                    session=sess,
                    updates=update,
                    encoder_pc={'inputs': input_ids, 'input_len': input_length},
                    code_pc=code_analyzer.placeholders,
                )
                saver.save(sess, 'models/pretrain_128_64_1', train_epoch)
                print(f'epoch {train_epoch} train {tr_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def main():
    with open('django_data_set_4.json') as f:
        data_set = json.load(f)
    descriptions = data_set['train']['indexed_description']
    trees = data_set['train']['ast_tree']
    nodes = convert_trees_to_node(trees, data_set['ast_token_index'][SUBTREE_START_TOKEN])
    zero_index = data_set['ast_token_index'][SEQUENCE_END_TOKEN]
    input_end_marker = data_set['desc_word_index'][SEQUENCE_END_TOKEN]
    code_num_tokens = len(data_set['ast_token_index'])
    desc_num_tokens = len(data_set['desc_word_index'])
    nodes = [prepare_sample(it, zero_index) for it in nodes]
    data_set = list(zip(descriptions, nodes))
    train(data_set, desc_num_tokens, code_num_tokens, input_end_marker)


if __name__ == '__main__':
    main()
