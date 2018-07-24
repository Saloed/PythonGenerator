from collections import OrderedDict

import numpy as np
import tensorflow as tf

LEARN_RATE = 0.07  # 0.001
L2_PARAM = 5.e-5
DROPOUT = 0.8
# NUM_EMBEDDINGS = 30
NUM_FEATURES = 30
NUM_CONVOLUTION = 200
NUM_HIDDEN = 128
SAVE_PERIOD = 20
BATCH_SIZE = 6
NUM_RETRY = 200
TOKEN_THRESHOLD = 150
RANDOM_RANGE = 0.02
NUM_EPOCH = 50


class Params:
    def __init__(self, weights, bias, embeddings):
        self.w = weights
        self.b = bias
        self.embeddings = embeddings


def rand_weight(shape_0, shape_1, name, params: dict):
    with tf.name_scope(name):
        var = tf.Variable(
            tf.random_uniform([shape_1, shape_0], -RANDOM_RANGE, RANDOM_RANGE),
            # tf.truncated_normal(shape=[shape_1, shape_0], stddev=RANDOM_RANGE),
            name=name)
        variable_summaries(var)
        params[name] = var
    return var


def rand_bias(shape, name, params: dict):
    return rand_weight(shape, 1, name, params)


def init_params(num_tokens):
    with tf.name_scope('Embeddings'):
        embeddings = tf.Variable(tf.random_uniform((num_tokens, NUM_FEATURES), -1, 1), dtype=tf.float32)

    with tf.name_scope('Params'):
        weights = {}
        # rand_weight(NUM_FEATURES, NUM_EMBEDDINGS, 'w_emb', weights)
        rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root', weights)
        rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left', weights)
        rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right', weights)
        rand_weight(NUM_HIDDEN, NUM_CONVOLUTION, 'w_hid', weights)
        # rand_weight(author_amount, NUM_HIDDEN, 'w_out', weights)
        bias = {}
        # rand_bias(NUM_FEATURES, 'b_emb', bias)
        rand_bias(NUM_CONVOLUTION, 'b_conv', bias)
        rand_bias(NUM_HIDDEN, 'b_hid', bias)
        # rand_bias(author_amount, 'b_out', bias)

    return Params(weights, bias, embeddings)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean, variance = tf.nn.moments(var, [1, 0])
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(variance)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)
