import json
from traceback import print_tb

import numpy as np
import tensorflow as tf

import batching
from current_net_conf import *
from model import model_full
from tbcnn.BatchBuilder import prepare_sample
from tbcnn.NetBuilder import build_net
from tbcnn.TFParameters import init_params
from tbcnn.prepare_ast_to_tree import convert_trees_to_node


def make_batcher(query_seq_end):
    def batching_function(batch):
        query = [q for q, _ in batch]
        code = [c for _, c in batch]
        query = batching.align_batch(query, query_seq_end)
        return query, code

    return batching_function


def make_fetches(model):
    return model[1]


def make_feed(data, model):
    (query, query_len), code = data
    (encoder_pc, code_pc) = model[0]
    pc = {
        encoder_pc['query_ids']: query,
        encoder_pc['query_length']: query_len,
    }
    for i in range(BATCH_SIZE):
        pc.update(code_pc[i].assign(code[i]))
    return pc


def feed_from_data_set(data_set, model):
    for batch in data_set:
        yield make_feed(batch, model)


def build_model(query_tokens_count, code_num_tokens):
    encoder, placeholders = model_full.build_encoder(query_tokens_count)

    code_analyzer_params = init_params(code_num_tokens)
    code_analyzer = build_net(code_analyzer_params)

    positive_net_out = code_analyzer.out
    negative_net_out = tf.random_shuffle(positive_net_out)
    negative_net_out = tf.stop_gradient(negative_net_out)
    positive_distance = tf.norm(encoder.last_state - positive_net_out, axis=1)
    negative_distance = tf.norm(encoder.last_state - negative_net_out, axis=1)
    loss = positive_distance + 1.0 - negative_distance
    loss = tf.nn.relu(loss)
    loss = tf.reduce_mean(loss)
    update = tf.train.AdamOptimizer(0.0005).minimize(loss)

    pc = placeholders, code_analyzer.placeholders
    fetches = [loss, update]
    return pc, fetches


def run_pretrain(
        data_set,
        session,
        model
):
    result_loss = []
    batch_count = len(data_set)
    fetches = make_fetches(model)
    for j, feed in enumerate(feed_from_data_set(data_set, model)):
        err, *_ = session.run(fetches=fetches, feed_dict=feed)
        result_loss.append(float(err))
        batch_number = j + 1
        if batch_number % 200 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')

    return np.mean(result_loss)


def train(data_set, model):
    encoder_variables = model_full.get_pretrained_variables()
    initializer = tf.global_variables_initializer()
    config = tf.ConfigProto()
    saver = tf.train.Saver(var_list=encoder_variables, max_to_keep=100)
    pretrain_model_name = PRETRAIN_SAVE_PATH + PRETRAIN_BASE_NAME + '_{}_{}_{}'.format(
        ENCODER_INPUT_SIZE, ENCODER_STATE_SIZE, ENCODER_LAYERS)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                tr_loss = run_pretrain(
                    data_set=data_set,
                    model=model,
                    session=sess,
                )
                saver.save(sess, pretrain_model_name, train_epoch)
                print(f'epoch {train_epoch} train {tr_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)


def main():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    train_set = data_set['train']
    queries = train_set['queries']
    trees = train_set['trees']
    nodes = convert_trees_to_node(trees)
    code_num_tokens = train_set['tree_size']
    num_query_tokens = data_set['train']['query_tokens_count']
    query_end_marker = data_set['train']['query_seq_end']

    zero_index = code_num_tokens
    code_num_tokens += 1
    nodes = [prepare_sample(it, zero_index) for it in nodes]

    data_set = list(zip(queries, nodes))

    batcher = make_batcher(query_end_marker)
    train_batches = batching.group_by_batches(data_set, batcher, sort_key=lambda x: len(x[0]))
    model = build_model(num_query_tokens, code_num_tokens)
    train(train_batches, model)


if __name__ == '__main__':
    main()
