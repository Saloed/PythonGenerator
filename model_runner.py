import collections
from traceback import print_tb

from tqdm import tqdm

import numpy as np
import tensorflow as tf

from model import model_full
from current_net_conf import *


def feed_from_data_set(data_set, model, feeder):
    for _id, *data in data_set:
        yield _id, feeder(data, model)


def stats_to_str(stats):
    results = ((name, np.mean(stat)) for name, stat in stats.items())
    result_strs = (f'{name}: {stat:.04f}' for name, stat in results)
    res_str = ' | '.join(result_strs)
    return res_str


def _run_model(data_set, model, session, is_train, stat_fetcher, stat_saver, feeder):
    fetches = model_full.make_fetches(model, is_train)
    stats_fetches = stat_fetcher(model)
    stats_border = len(stats_fetches)
    stats = collections.defaultdict(list)
    fetches = stats_fetches + fetches
    result_loss = []
    for ids, feed in feed_from_data_set(data_set, model, feeder):
        results = session.run(fetches=fetches, feed_dict=feed)
        stats_result, other_result = results[:stats_border], results[stats_border:]
        loss = other_result[0]
        result_loss.append(float(loss))
        stat_saver(stats_result, stats)

    stats['loss'] = result_loss
    stats_str = stats_to_str(stats)

    return stats_str


def train_model(
        train_data_set,
        valid_data_set,
        model,
        is_rules_model
):
    if is_rules_model:
        feeder = model_full.make_rules_feed
        stat_fetcher = model_full.make_rules_stats_fetches
        stat_saver = model_full.update_rules_stats
    else:
        feeder = model_full.make_words_feed
        stat_fetcher = model_full.make_words_stats_fetches
        stat_saver = model_full.update_words_stats

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    initializer = tf.global_variables_initializer()
    local_initializer = tf.local_variables_initializer()

    if is_rules_model:
        encoder_variables = model_full.get_pretrained_variables()
        encoder_variables_restorer = tf.train.Saver(var_list=encoder_variables)
        pretrain_model_name = PRETRAIN_SAVE_PATH + PRETRAIN_BASE_NAME + '_{}_{}_{}'.format(
            RULES_QUERY_ENCODER_INPUT_SIZE, RULES_QUERY_ENCODER_STATE_SIZE, RULES_QUERY_ENCODER_LAYERS)
    # else:
    #     pretrain_model_name = MODEL_SAVE_PATH + BEST_RULES_MODEL_BASE_NAME

    model_name = MODEL_SAVE_PATH + (RULES_MODEL_BASE_NAME if is_rules_model else WORDS_MODEL_BASE_NAME)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        graph_writer = tf.summary.FileWriter('tmp', sess.graph)

        sess.run(initializer)
        sess.run(local_initializer)
        if is_rules_model:
            encoder_variables_restorer.restore(sess, pretrain_model_name)

        try:
            for train_epoch in range(1000):
                tr_stats = _run_model(
                    data_set=tqdm(train_data_set, desc=f'train {train_epoch}'),
                    model=model,
                    session=sess,
                    is_train=True,
                    stat_fetcher=stat_fetcher,
                    stat_saver=stat_saver,
                    feeder=feeder
                )
                v_stats = _run_model(
                    data_set=tqdm(valid_data_set, desc=f'valid {train_epoch}'),
                    model=model,
                    session=sess,
                    is_train=False,
                    stat_fetcher=stat_fetcher,
                    stat_saver=stat_saver,
                    feeder=feeder
                )
                saver.save(sess, model_name, train_epoch)
                print(f'train: {tr_stats} | valid: {v_stats}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)
