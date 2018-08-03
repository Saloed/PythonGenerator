import collections
from traceback import print_tb

import numpy as np
import tensorflow as tf

from model import model_full
from current_net_conf import *


def feed_from_data_set(data_set, model):
    for _id, *data in data_set:
        yield _id, model_full.make_feed(data, model)


def print_stats(stats):
    results = ((name, np.mean(stat)) for name, stat in stats.items())
    result_strs = ('{}: {}'.format(name, stat) for name, stat in results)
    res_str = '\n'.join(result_strs)
    print(res_str)


def _run_model(data_set, model, session, is_train):
    fetches = model_full.make_fetches(model, is_train)
    stats_fetches = model_full.make_stats_fetches(model)
    stats_border = len(stats_fetches)
    stats = collections.defaultdict(list)
    fetches = stats_fetches + fetches
    result_loss = []
    batch_count = len(data_set)
    for j, (ids, feed) in enumerate(feed_from_data_set(data_set, model)):
        results = session.run(fetches=fetches, feed_dict=feed)
        stats_result, other_result = results[:stats_border], results[stats_border:]
        err = other_result[0]
        result_loss.append(float(err))
        model_full.update_stats(stats_result, stats)
        batch_number = j + 1
        if batch_number % 200 == 0:
            percent = int(j / batch_count * 100)
            print(f'Complete {percent}')
            print_stats(stats)
    print('Epoch stats')
    print_stats(stats)

    return np.mean(result_loss)


def train_model(
        train_data_set,
        valid_data_set,
        model,
):
    encoder_variables = model_full.get_pretrained_variables()

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    initializer = tf.global_variables_initializer()
    local_initializer = tf.local_variables_initializer()

    encoder_variables_restorer = tf.train.Saver(var_list=encoder_variables)

    pretrain_model_name = PRETRAIN_SAVE_PATH + PRETRAIN_BASE_NAME + '_{}_{}_{}'.format(
        ENCODER_INPUT_SIZE, ENCODER_STATE_SIZE, ENCODER_LAYERS)

    model_name = MODEL_SAVE_PATH + MODEL_BASE_NAME

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        sess.run(initializer)
        sess.run(local_initializer)
        encoder_variables_restorer.restore(sess, pretrain_model_name)
        try:
            for train_epoch in range(1000):
                print(f'start epoch {train_epoch}')

                tr_loss = _run_model(
                    data_set=train_data_set,
                    model=model,
                    session=sess,
                    is_train=True,
                )
                print('valid epoch')
                v_loss = _run_model(
                    data_set=valid_data_set,
                    model=model,
                    session=sess,
                    is_train=False,
                )
                saver.save(sess, model_name, train_epoch)
                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:

            print(ex)
            print_tb(ex.__traceback__)
