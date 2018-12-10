import json
import random

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from batching import align_batch
from model.selector import build_selector, build_selector_loss
from current_net_conf import *


def run_subset(session, data_set, feeder, accuracy, updates=None, name=None):
    stats = []
    for sample in tqdm(data_set, name):
        feed = feeder(sample)
        fetches = [accuracy]
        if updates is not None:
            fetches += [updates]
        res = session.run(fetches, feed)
        acc = res[0]
        stats.append(acc)
    return np.mean(stats)


def make_feeder(placeholders, rules_end_marker, nodes_end_marker):
    def feeder(sample):
        sample_id, query, true_data_id, data = sample
        _rules = [rule for rule, _, _ in data]
        _nodes = [node for _, node, _ in data]
        _parent_rules = [pr for _, _, pr in data]

        rules, rules_length = align_batch(_rules, rules_end_marker, need_length=True)
        parent_rules = align_batch(_parent_rules, rules_end_marker, need_length=False)
        nodes = align_batch(_nodes, nodes_end_marker, need_length=False)

        feed = placeholders.feed(query, rules, rules_length, nodes, parent_rules)
        feed[placeholders.scores_target] = true_data_id

        return feed

    return feeder


def train_selector():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME + '_selector') as f:
        data_set = json.load(f)

    num_rules = data_set['train']['rules_count']
    num_query_tokens = data_set['train']['query_count']
    num_nodes = data_set['train']['nodes_count'] + 1  # fixme: aaaaa fix this

    query_end_marker = data_set['train']['query_end_marker']
    rules_end_marker = data_set['train']['rules_end_marker']
    nodes_end_marker = data_set['train']['nodes_end_marker']

    train_data = data_set['train']['data']
    valid_data = data_set['valid']['data']

    selector, placeholders = build_selector(num_query_tokens, num_rules, num_nodes)
    l2_loss, accuracy = build_selector_loss(selector, placeholders)

    feeder = make_feeder(placeholders, rules_end_marker, nodes_end_marker)

    # optimizer = tf.train.RMSPropOptimizer(0.001)
    optimizer = tf.train.AdamOptimizer(0.0001)
    updates = optimizer.minimize(l2_loss)

    model_name = MODEL_SAVE_PATH + SELECTOR_MODEL_BASE_NAME
    saver = tf.train.Saver(max_to_keep=100)

    initializer = tf.global_variables_initializer()

    # train_data = train_data[:10000]  # todo: remove it

    with tf.Session() as sess:
        sess.run(initializer)
        for epoch in range(100):
            random.shuffle(train_data)
            train_acc = run_subset(sess, train_data, feeder, accuracy, updates, 'train')
            valid_acc = run_subset(sess, valid_data, feeder, accuracy, name='valid')
            print(f'epoch {epoch:2d} | train: {train_acc:.04f} | valid: {valid_acc:.04f}')
            saver.save(sess, model_name, epoch)


if __name__ == '__main__':
    train_selector()
