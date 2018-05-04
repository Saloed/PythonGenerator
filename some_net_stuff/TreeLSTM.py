import tensorflow as tf
from .batch_tree import BatchTree, BatchTreeSample
import numpy as np
from sklearn import metrics
import collections

import sys

from .cool_stuff import extract_tree_data, extract_batch_tree_data


class NarytreeLSTM(object):
    def __init__(self, config=None):
        self.config = config

        with tf.variable_scope("Embed", regularizer=None):

            if config.embeddings is not None:
                initializer = config.embeddings
            else:
                initializer = tf.random_uniform((config.num_emb, config.emb_dim))
            self.embedding = tf.Variable(initial_value=initializer, trainable=config.trainable_embeddings,
                                         dtype='float32')

        with tf.variable_scope("Node",
                               initializer=
                               # tf.ones_initializer(),
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):

            def calc_wt_init(self, fan_in=300):
                eps = 1.0 / np.sqrt(fan_in)
                return eps

            self.U = tf.get_variable("U", [config.hidden_dim * config.degree, config.hidden_dim * (3 + config.degree)],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(config.hidden_dim)))
            self.W = tf.get_variable("W", [config.emb_dim, config.hidden_dim],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),
                                                                               calc_wt_init(config.emb_dim)))
            self.b = tf.get_variable("b", [config.hidden_dim * 3],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(
                                                                                   config.hidden_dim)))  # , regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.bf = tf.get_variable("bf", [config.hidden_dim],
                                      initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                                calc_wt_init(
                                                                                    config.hidden_dim)))  # , regularizer=tf.contrib.layers.l2_regularizer(0.0))

            self.observables = tf.placeholder(tf.int32, shape=[None])
            self.flows = tf.placeholder(tf.int32, shape=[None])
            self.input_scatter = tf.placeholder(tf.int32, shape=[None])
            self.observables_indices = tf.placeholder(tf.int32, shape=[None])
            self.out_indices = tf.placeholder(tf.int32, shape=[None])
            self.scatter_out = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in_indices = tf.placeholder(tf.int32, shape=[None])
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.tree_height = tf.placeholder(tf.int32, shape=[])
            self.dropout = tf.placeholder(tf.float32, shape=[])
            self.child_scatter_indices = tf.placeholder(tf.int32, shape=[None])
            self.nodes_count = tf.placeholder(tf.int32, shape=[None])
            self.input_embed = tf.nn.embedding_lookup(self.embedding, self.observables)
            self.nodes_count_per_indice = tf.placeholder(tf.float32, shape=[None])

            self.training_variables = [self.U, self.W, self.b, self.bf]
            if config.trainable_embeddings:
                self.training_variables.append(self.embedding)

    def get_feed_dict(self, batch_sample, dropout=1.0):
        # print batch_sample.scatter_in
        # print batch_sample.scatter_in_indices
        # print batch_sample.nodes_count_per_indice, "nodes_count_per_indice"
        return {
            self.observables: batch_sample.observables,
            self.flows: batch_sample.flows,
            self.input_scatter: batch_sample.input_scatter,
            self.observables_indices: batch_sample.observables_indices,
            self.out_indices: batch_sample.out_indices,
            self.tree_height: len(batch_sample.out_indices) - 1,
            self.batch_size: batch_sample.flows[-1],  # batch_sample.out_indices[-1] - batch_sample.out_indices[-2],
            self.scatter_out: batch_sample.scatter_out,
            self.scatter_in: batch_sample.scatter_in,
            self.scatter_in_indices: batch_sample.scatter_in_indices,
            self.child_scatter_indices: batch_sample.child_scatter_indices,
            self.nodes_count: batch_sample.nodes_count,
            self.dropout: dropout,
            self.nodes_count_per_indice: batch_sample.nodes_count_per_indice
        }

    def get_output(self):
        nodes_h, _ = self.get_outputs()
        return nodes_h

    def get_output_unscattered(self):
        _, nodes_h_unscattered = self.get_outputs()
        return nodes_h_unscattered

    def get_outputs(self):
        with tf.variable_scope("Node", reuse=True):
            W = tf.get_variable("W", [self.config.emb_dim, self.config.hidden_dim])
            U = tf.get_variable("U", [self.config.hidden_dim * self.config.degree,
                                      self.config.hidden_dim * (3 + self.config.degree)])
            b = tf.get_variable("b", [3 * self.config.hidden_dim])
            bf = tf.get_variable("bf", [self.config.hidden_dim])

            nbf = tf.tile(bf, [self.config.degree])

            nodes_h_scattered = tf.TensorArray(tf.float32, size=self.tree_height, clear_after_read=False)
            nodes_h = tf.TensorArray(tf.float32, size=self.tree_height, clear_after_read=False)
            nodes_c = tf.TensorArray(tf.float32, size=self.tree_height, clear_after_read=False)

            const0f = tf.constant([0], dtype=tf.float32)
            idx_var = tf.constant(0, dtype=tf.int32)
            hidden_shape = tf.constant([-1, self.config.hidden_dim * self.config.degree], dtype=tf.int32)
            out_shape = tf.stack([-1, self.batch_size, self.config.hidden_dim], 0)

            def _recurrence(nodes_h, nodes_c, nodes_h_scattered, idx_var):
                out_ = tf.concat([nbf, b], axis=0)
                idx_var_dim1 = tf.expand_dims(idx_var, 0)
                prev_idx_var_dim1 = tf.expand_dims(idx_var - 1, 0)

                observables_indice_begin, observables_indice_end = tf.split(
                    tf.slice(self.observables_indices, idx_var_dim1, [2]), 2)
                observables_size = observables_indice_end - observables_indice_begin
                out_indice_begin, out_indice_end = tf.split(
                    tf.slice(self.out_indices, idx_var_dim1, [2]), 2)
                out_size = out_indice_end - out_indice_begin
                flow = tf.slice(self.flows, idx_var_dim1, [1])
                w_scatter_shape = tf.concat([flow, [self.config.hidden_dim]], axis=0)
                u_scatter_shape = tf.concat([flow, [self.config.hidden_dim * (3 + self.config.degree)]], axis=0)
                c_scatter_shape = tf.concat([flow, [self.config.hidden_dim * self.config.degree]], axis=0)

                def compute_indices():
                    prev_level_indice_begin, prev_level_indice_end = tf.split(
                        tf.slice(self.out_indices, prev_idx_var_dim1, [2]), 2)
                    prev_level_indice_size = prev_level_indice_end - prev_level_indice_begin
                    scatter_indice_begin, scatter_indice_end = tf.split(
                        tf.slice(self.scatter_in_indices, prev_idx_var_dim1, [2]), 2)
                    scatter_indice_size = scatter_indice_end - scatter_indice_begin
                    child_scatters = tf.slice(self.child_scatter_indices, prev_level_indice_begin,
                                              prev_level_indice_size)
                    child_scatters = tf.reshape(child_scatters, tf.concat([prev_level_indice_size, [-1]], 0))
                    return scatter_indice_begin, scatter_indice_size, child_scatters

                def hs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    h = nodes_h.read(idx_var - 1)
                    hs = tf.scatter_nd(child_scatters, h, tf.shape(h), name=None)
                    hs = tf.reshape(hs, hidden_shape)
                    out = tf.matmul(hs, U)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    # scatters_in = tf.Print(scatters_in, [idx_var, tf.shape(hs), u_scatter_shape, scatters_in], "hs", 300, 300)
                    out = tf.scatter_nd(scatters_in, out, u_scatter_shape, name=None)
                    return out

                def cs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    c = nodes_c.read(idx_var - 1)
                    cs = tf.scatter_nd(child_scatters, c, tf.shape(c), name=None)
                    cs = tf.reshape(cs, hidden_shape)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    # scatters_in = tf.Print(scatters_in, [idx_var, tf.shape(cs), c_scatter_shape, scatters_in], "cs",
                    #                       300, 300)
                    cs = tf.scatter_nd(scatters_in, cs, c_scatter_shape, name=None)
                    return cs

                out_ += tf.cond(tf.less(0, idx_var),
                                lambda: hs_compute(),
                                lambda: const0f
                                )
                cs = tf.cond(tf.less(0, idx_var),
                             lambda: cs_compute(),
                             lambda: const0f
                             )

                observable = tf.squeeze(tf.slice(self.observables, observables_indice_begin, observables_size))

                input_embed = tf.reshape(tf.nn.embedding_lookup(self.embedding, observable), [-1, self.config.emb_dim])

                def compute_input():
                    out = tf.matmul(input_embed, W)

                    input_scatter = tf.slice(self.input_scatter, observables_indice_begin, observables_size)
                    input_scatter = tf.reshape(input_scatter, tf.concat([observables_size, [-1]], 0))
                    out = tf.scatter_nd(input_scatter, out, w_scatter_shape, name=None)
                    out = tf.tile(out, [1, 3 + self.config.degree])
                    return out

                out_ += tf.cond(tf.less(0, tf.squeeze(observables_size)),
                                lambda: compute_input(),
                                lambda: const0f)

                v = tf.split(out_, 3 + self.config.degree, axis=1)
                vf = tf.sigmoid(tf.concat(v[:self.config.degree], axis=1))

                c = tf.cond(tf.less(0, idx_var),
                            lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),
                                                tf.tanh(v[self.config.degree + 2])) + tf.reduce_sum(
                                tf.stack(tf.split(tf.multiply(vf, cs), self.config.degree, axis=1)), axis=0),
                            lambda: tf.multiply(tf.sigmoid(v[self.config.degree]), tf.tanh(v[self.config.degree + 2]))
                            )

                h = tf.multiply(tf.sigmoid(v[self.config.degree + 1]), tf.tanh(c))
                h = tf.nn.dropout(h, self.dropout)
                slice = tf.slice(self.embedding, [32, 0], [1, 10])
                # h = tf.Print(h, [slice], "the DOT embed", 300, 300)
                nodes_h = nodes_h.write(idx_var, h)
                nodes_c = nodes_c.write(idx_var, c)

                scatters = tf.reshape(tf.slice(self.scatter_out, out_indice_begin, out_size),
                                      tf.concat([out_size, [-1]], 0))

                node_count = tf.slice(self.nodes_count, idx_var_dim1, [1])
                scatter_out_lenght = node_count * self.batch_size
                scatter_out_shape = tf.stack([tf.squeeze(scatter_out_lenght), self.config.hidden_dim], 0)
                h = tf.reshape(tf.scatter_nd(scatters, h, scatter_out_shape, name=None), out_shape)
                nodes_h_scattered = nodes_h_scattered.write(idx_var, h)
                idx_var = tf.add(idx_var, 1)

                return nodes_h, nodes_c, nodes_h_scattered, idx_var

            loop_cond = lambda x, y, z, id: tf.less(id, self.tree_height)

            loop_vars = [nodes_h, nodes_c, nodes_h_scattered, idx_var]
            nodes_h, nodes_c, nodes_h_scattered, idx_var = tf.while_loop(loop_cond, _recurrence, loop_vars,
                                                                         parallel_iterations=1)
            return nodes_h_scattered.concat(), nodes_h


class SoftMaxNarytreeLSTM(object):

    def __init__(self, config, data):
        def calc_wt_init(self, fan_in=300):
            eps = 1.0 / np.sqrt(fan_in)
            return eps

        self.config = config
        with tf.variable_scope("Predictor",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):
            self.tree_lstm = NarytreeLSTM(config)
            self.W = tf.get_variable("W", [config.hidden_dim, config.num_labels],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(config.hidden_dim)))
            self.b = tf.get_variable("b", [config.num_labels],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(
                                                                                   config.hidden_dim)))  # , regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.training_variables = [self.W, self.b] + self.tree_lstm.training_variables
            self.optimizer = tf.train.AdagradOptimizer(self.config.lr)
            self.embed_optimizer = tf.train.AdagradOptimizer(self.config.emb_lr)
            self.loss = self.get_loss()
            # self.gv = self.optimizer.compute_gradients(self.loss, self.training_variables)
            self.gv = zip(tf.gradients(self.loss, self.training_variables), self.training_variables)
            if config.trainable_embeddings:
                self.opt = self.optimizer.apply_gradients(self.gv[:-1])
                self.embed_opt = self.embed_optimizer.apply_gradients(self.gv[-1:])
            else:
                self.opt = self.optimizer.apply_gradients(self.gv)
                self.embed_opt = tf.no_op()

            self.output = self.get_root_output()

    def get_root_output(self):
        nodes_h = self.tree_lstm.get_output_unscattered()
        roots_h = nodes_h.read(nodes_h.size() - 1)
        out = tf.matmul(roots_h, self.W) + self.b
        return out

    def get_output(self):
        return self.output

    def get_loss(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        # regpart = tf.Print(regpart, [regpart])
        h = self.tree_lstm.get_output_unscattered().concat()
        out = tf.matmul(h, self.W) + self.b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=out)
        return tf.reduce_sum(tf.divide(loss, tf.to_float(self.tree_lstm.batch_size))) + regpart

    def train(self, batch_tree, batch_labels, session):

        feed_dict = {self.labels: batch_tree.labels}
        feed_dict.update(self.tree_lstm.get_feed_dict(batch_tree, self.config.dropout))
        ce, _, _ = session.run([self.loss, self.opt, self.embed_opt], feed_dict=feed_dict)
        # v = session.run([self.output], feed_dict=feed_dict)
        # print("cross_entropy " + str(ce))
        return ce
        # print v

    def train_epoch(self, data, session):
        # from random import shuffle
        # shuffle(data)
        total_error = 0.0
        for batch in data:
            total_error += self.train(batch[0], batch[1], session)
        print('average error :', total_error / len(data))

    def test(self, data, session):
        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = tf.argmax(self.get_output(), 1)
            y_true = self.labels
            feed_dict = {self.labels: batch[0].root_labels}
            feed_dict.update(self.tree_lstm.get_feed_dict(batch[0]))
            y_pred, y_true = session.run([y_pred, y_true], feed_dict=feed_dict)
            ys_true += y_true.tolist()
            ys_pred += y_pred.tolist()
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print("Accuracy", score)
        # print "Recall", metrics.recall_score(ys_true, ys_pred)
        # print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print("confusion_matrix")
        print(metrics.confusion_matrix(ys_true, ys_pred))
        return score


class tf_NarytreeLSTM(object):

    def __init__(self, config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config = config
        self.batch_size = config.batch_size
        self.reg = self.config.reg
        self.degree = config.degree
        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()

        emb_leaves = self.add_embedding(config.embeddings)

        self.add_model_variables()

        self.batch_loss = self.compute_loss(emb_leaves)

        self.loss, self.total_loss = self.calc_batch_loss(self.batch_loss)

        self.train_op1, self.train_op2 = self.add_training_op()
        # self.train_op=tf.no_op()

    def add_embedding(self, embeddings):

        # embed=np.load('glove{0}_uniform.npy'.format(self.emb_dim))
        if embeddings is not None:
            initializer = embeddings
        else:
            initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope("Embed", regularizer=None):
            embedding = tf.Variable(initial_value=initializer, trainable=True, name='embedding', dtype='float32')
            ix = tf.to_int32(tf.not_equal(self.input, -1)) * self.input
            emb_tree = tf.nn.embedding_lookup(embedding, ix)
            emb_tree = emb_tree * (tf.expand_dims(
                tf.to_float(tf.not_equal(self.input, -1)), 2))

            return emb_tree

    def add_placeholders(self):
        dim2 = self.config.maxnodesize
        dim1 = self.config.batch_size
        self.input = tf.placeholder(tf.int32, [dim1, dim2], name='input')
        self.treestr = tf.placeholder(tf.int32, [dim1, dim2, 2], name='tree')
        self.labels = tf.placeholder(tf.int32, [dim1, dim2], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr, -1)), [1, 2])
        self.n_inodes = self.n_inodes / 2

        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input, -1)), [1])
        self.batch_len = tf.placeholder(tf.int32, name="batch_len")

    def calc_wt_init(self, fan_in=300):
        eps = 1.0 / np.sqrt(fan_in)
        return eps

    def add_model_variables(self):

        with tf.variable_scope("Composition",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=
                               tf.contrib.layers.l2_regularizer(self.config.reg
                                                                )):
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(), self.calc_wt_init()))
            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),
                                                                           self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb", [4 * self.hidden_dim], initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))
            # cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim])
            # cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim])
            # cb = tf.get_variable("cb",[4*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
        with tf.variable_scope("Projection", regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),
                                                                          self.calc_wt_init(self.hidden_dim))
                                )
            bu = tf.get_variable("bu", [self.output_dim], initializer=
            tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self, emb):

        with tf.variable_scope("Composition", reuse=True):
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            b = tf.slice(cb, [0], [2 * self.hidden_dim])

            def _recurseleaf(x):
                concat_uo = tf.matmul(tf.expand_dims(x, 0), cU) + b
                u, o = tf.split(axis=1, num_or_size_splits=2, value=concat_uo)
                o = tf.nn.sigmoid(o)
                u = tf.nn.tanh(u)

                c = u  # tf.squeeze(u)
                h = o * tf.nn.tanh(c)

                hc = tf.concat(axis=1, values=[h, c])
                hc = tf.squeeze(hc)
                return hc

        hc = tf.map_fn(_recurseleaf, emb)
        return hc

    def compute_loss(self, emb_batch, curr_batch_size=None):
        outloss = []
        prediction = []
        for idx_batch in range(self.config.batch_size):
            tree_states = self.compute_states(emb_batch, idx_batch)
            logits = self.create_output(tree_states)

            labels1 = tf.gather(self.labels, idx_batch)
            labels2 = tf.reduce_sum(tf.to_int32(tf.not_equal(labels1, -1)))
            labels = tf.gather(labels1, tf.range(labels2))
            loss = self.calc_loss(logits, labels)

            pred = tf.nn.softmax(logits)

            pred_root = tf.gather(pred, labels2 - 1)

            prediction.append(pred_root)
            outloss.append(loss)

        batch_loss = tf.stack(outloss)
        self.pred = tf.stack(prediction)

        return batch_loss

    def compute_states(self, emb, idx_batch=0):

        # nb of leaves in this sample
        num_leaves = tf.squeeze(tf.gather(self.num_leaves, idx_batch))
        # num_leaves=tf.Print(num_leaves,[num_leaves])
        n_inodes = tf.gather(self.n_inodes, idx_batch)
        # embx=tf.gather(emb,tf.range(num_leaves))
        embx = tf.gather(tf.gather(emb, idx_batch), tf.range(num_leaves))
        # treestr=self.treestr#tf.gather(self.treestr,tf.range(self.n_inodes))
        treestr = tf.gather(tf.gather(self.treestr, idx_batch), tf.range(n_inodes))
        leaf_hc = self.process_leafs(embx)
        leaf_h, leaf_c = tf.split(axis=1, num_or_size_splits=2, value=leaf_hc)

        node_h = tf.identity(leaf_h)
        node_c = tf.identity(leaf_c)

        idx_var = tf.constant(0)  # tf.Variable(0,trainable=False)

        with tf.variable_scope("Composition", reuse=True):
            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            bu, bo, bi, bf = tf.split(axis=0, num_or_size_splits=4, value=cb)

            def _recurrence(node_h, node_c, idx_var):
                node_info = tf.gather(treestr, idx_var)

                child_h = tf.gather(node_h, node_info)
                child_c = tf.gather(node_c, node_info)

                flat_ = tf.reshape(child_h, [-1])
                tmp = tf.matmul(tf.expand_dims(flat_, 0), cW)
                u, o, i, fl, fr = tf.split(axis=1, num_or_size_splits=5, value=tmp)

                i = tf.nn.sigmoid(i + bi)
                o = tf.nn.sigmoid(o + bo)
                u = tf.nn.tanh(u + bu)
                fl = tf.nn.sigmoid(fl + bf)
                fr = tf.nn.sigmoid(fr + bf)

                f = tf.concat(axis=0, values=[fl, fr])
                c = i * u + tf.reduce_sum(f * child_c, [0])
                h = o * tf.nn.tanh(c)

                node_h = tf.concat(axis=0, values=[node_h, h])

                node_c = tf.concat(axis=0, values=[node_c, c])

                idx_var = tf.add(idx_var, 1)

                return node_h, node_c, idx_var

            loop_cond = lambda a1, b1, idx_var: tf.less(idx_var, n_inodes)

            loop_vars = [node_h, node_c, idx_var]
            node_h, node_c, idx_var = tf.while_loop(loop_cond, _recurrence,
                                                    loop_vars, parallel_iterations=10)

            return node_h

    def create_output(self, tree_states):

        with tf.variable_scope("Projection", reuse=True):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                )
            bu = tf.get_variable("bu", [self.output_dim])

            h = tf.matmul(tree_states, U, transpose_b=True) + bu
            return h

    def calc_loss(self, logits, labels):

        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_sum(l1, [0])
        return loss

    def calc_batch_loss(self, batch_loss):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        loss = tf.reduce_mean(batch_loss)
        total_loss = loss + 0.5 * regpart
        return loss, total_loss

    def add_training_op_old(self):

        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op

    def add_training_op(self):
        loss = self.total_loss
        opt1 = tf.train.AdagradOptimizer(self.config.lr)
        opt2 = tf.train.AdagradOptimizer(self.config.emb_lr)

        ts = tf.trainable_variables()
        gs = tf.gradients(loss, ts)
        gs_ts = zip(gs, ts)

        gt_emb, gt_nn = [], []
        for g, t in gs_ts:
            # print t.name,g.name
            if "Embed/embedding:0" in t.name:
                # g=tf.Print(g,[g.get_shape(),t.get_shape()])
                gt_emb.append((g, t))
                # print t.name
            else:
                gt_nn.append((g, t))
                # print t.name

        train_op1 = opt1.apply_gradients(gt_nn)
        train_op2 = opt2.apply_gradients(gt_emb)
        train_op = [train_op1, train_op2]

        return train_op

    def train(self, data, sess):
        from random import shuffle
        data_idxs = range(len(data))
        data_idxs.reverse()
        # shuffle(data_idxs)
        losses = []
        for i in range(0, len(data), self.batch_size):
            batch_size = min(i + self.batch_size, len(data)) - i
            if batch_size < self.batch_size: break

            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [data[ix] for ix in batch_idxs]  # [i:i+batch_size]

            input_b, treestr_b, labels_b = extract_batch_tree_data(batch_data, self.config.maxnodesize)

            feed = {self.input: input_b, self.treestr: treestr_b, self.labels: labels_b,
                    self.dropout: self.config.dropout, self.batch_len: len(input_b)}

            loss, bloss, _, _ = sess.run([self.loss, self.batch_loss, self.train_op1, self.train_op2], feed_dict=feed)
            # sess.run(self.train_op,feed_dict=feed)
            # print np.mean(bloss)
            losses.append(loss)
            avg_loss = np.mean(losses)
            sstr = 'avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            sys.stdout.write(sstr)
            sys.stdout.flush()

            # if i>1000: break
        return np.mean(losses)

    def evaluate(self, data, sess):
        num_correct = 0
        total_data = 0
        data_idxs = range(len(data))
        test_batch_size = self.config.batch_size
        losses = []
        for i in range(0, len(data), test_batch_size):
            batch_size = min(i + test_batch_size, len(data)) - i
            if batch_size < test_batch_size: break
            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [data[ix] for ix in batch_idxs]  # [i:i+batch_size]
            labels_root = [l for _, l in batch_data]
            input_b, treestr_b, labels_b = extract_batch_tree_data(batch_data, self.config.maxnodesize)

            feed = {self.input: input_b, self.treestr: treestr_b, self.labels: labels_b, self.dropout: 1.0,
                    self.batch_len: len(input_b)}

            pred_y = sess.run(self.pred, feed_dict=feed)
            # print pred_y,labels_root
            y = np.argmax(pred_y, axis=1)
            # num_correct+=np.sum(y==np.array(labels_root))
            for i, v in enumerate(labels_root):
                if y[i] == v: num_correct += 1
                total_data += 1
            # break
        print("total_data", total_data)
        print("num_correct", num_correct)
        acc = float(num_correct) / float(total_data)
        return acc


class tf_ChildsumtreeLSTM(tf_NarytreeLSTM):

    def add_model_variables(self):
        with tf.variable_scope("Composition",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=
                               tf.contrib.layers.l2_regularizer(self.config.reg
                                                                )):
            cUW = tf.get_variable("cUW", [self.emb_dim + self.hidden_dim, 4 * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim], initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

        with tf.variable_scope("Projection", regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                initializer=tf.random_uniform_initializer(
                                    -0.05, 0.05))
            bu = tf.get_variable("bu", [self.output_dim], initializer=
            tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self, emb):
        with tf.variable_scope("Composition", reuse=True):
            cUW = tf.get_variable("cUW")
            cb = tf.get_variable("cb")
            U = tf.slice(cUW, [0, 0], [self.emb_dim, 2 * self.hidden_dim])
            b = tf.slice(cb, [0], [2 * self.hidden_dim])

            def _recurseleaf(x):
                concat_uo = tf.matmul(tf.expand_dims(x, 0), U) + b
                u, o = tf.split(axis=1, num_or_size_splits=2, value=concat_uo)
                o = tf.nn.sigmoid(o)
                u = tf.nn.tanh(u)

                c = u  # tf.squeeze(u)
                h = o * tf.nn.tanh(c)

                hc = tf.concat(axis=1, values=[h, c])
                hc = tf.squeeze(hc)
                return hc

            hc = tf.map_fn(_recurseleaf, emb)
            return hc

    def compute_states(self, emb, idx_batch=0):
        # if num_leaves is None:
        # num_leaves = self.n_nodes - self.n_inodes
        num_leaves = tf.squeeze(tf.gather(self.num_leaves, idx_batch))
        # num_leaves=tf.Print(num_leaves,[num_leaves])
        n_inodes = tf.gather(self.n_inodes, idx_batch)
        # embx=tf.gather(emb,tf.range(num_leaves))
        emb_tree = tf.gather(emb, idx_batch)
        emb_leaf = tf.gather(emb_tree, tf.range(num_leaves))
        # treestr=self.treestr#tf.gather(self.treestr,tf.range(self.n_inodes))
        treestr = tf.gather(tf.gather(self.treestr, idx_batch), tf.range(n_inodes))
        leaf_hc = self.process_leafs(emb_leaf)
        leaf_h, leaf_c = tf.split(axis=1, num_or_size_splits=2, value=leaf_hc)

        node_h = tf.identity(leaf_h)
        node_c = tf.identity(leaf_c)

        idx_var = tf.constant(0)  # tf.Variable(0,trainable=False)

        with tf.variable_scope("Composition", reuse=True):
            cUW = tf.get_variable("cUW", [self.emb_dim + self.hidden_dim, 4 * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            bu, bo, bi, bf = tf.split(axis=0, num_or_size_splits=4, value=cb)

            UW = tf.slice(cUW, [0, 0], [-1, 3 * self.hidden_dim])

            U_fW_f = tf.slice(cUW, [0, 3 * self.hidden_dim], [-1, -1])

            def _recurrence(emb_tree, node_h, node_c, idx_var):
                node_x = tf.gather(emb_tree, num_leaves + idx_var)
                # node_x=tf.zeros([self.emb_dim])
                node_info = tf.gather(treestr, idx_var)

                child_h = tf.gather(node_h, node_info)
                child_c = tf.gather(node_c, node_info)

                concat_xh = tf.concat(axis=0, values=[node_x, tf.reduce_sum(node_h, [0])])

                tmp = tf.matmul(tf.expand_dims(concat_xh, 0), UW)
                u, o, i = tf.split(value=1, num_or_size_splits=3, axis=tmp)
                # node_x=tf.Print(node_x,[tf.shape(node_x),node_x.get_shape()])
                hl, hr = tf.split(value=0, num_or_size_splits=2, axis=child_h)
                x_hl = tf.concat(axis=0, values=[node_x, tf.squeeze(hl)])
                x_hr = tf.concat(axis=0, values=[node_x, tf.squeeze(hr)])
                fl = tf.matmul(tf.expand_dims(x_hl, 0), U_fW_f)
                fr = tf.matmul(tf.expand_dims(x_hr, 0), U_fW_f)

                i = tf.nn.sigmoid(i + bi)
                o = tf.nn.sigmoid(o + bo)
                u = tf.nn.tanh(u + bu)
                fl = tf.nn.sigmoid(fl + bf)
                fr = tf.nn.sigmoid(fr + bf)

                f = tf.concat(axis=0, values=[fl, fr])
                c = i * u + tf.reduce_sum(f * child_c, [0])
                h = o * tf.nn.tanh(c)

                node_h = tf.concat(axis=0, values=[node_h, h])

                node_c = tf.concat(axis=0, values=[node_c, c])

                idx_var = tf.add(idx_var, 1)

                return emb_tree, node_h, node_c, idx_var

            loop_cond = lambda a1, b1, c1, idx_var: tf.less(idx_var, n_inodes)

            loop_vars = [emb_tree, node_h, node_c, idx_var]
            emb_tree, node_h, node_c, idx_var = tf.while_loop(loop_cond, _recurrence, loop_vars, parallel_iterations=1)

            return node_h


def test_lstm_model():
    class Config(object):
        num_emb = 10
        emb_dim = 3
        hidden_dim = 4
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 1.0
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = False
        embeddings = None
        batch_size = 7

    tree = BatchTree.empty_tree()
    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(1, 1, 1)
    tree.root.children[0].expand_or_add_child(1, 0, 0)
    tree.root.children[0].expand_or_add_child(1, 0, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(2, 1, 0)
    tree.root.expand_or_add_child(2, 1, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(3, 1, 1)
    tree.root.children[0].expand_or_add_child(3, 0, 0)
    tree.root.children[0].expand_or_add_child(3, 0, 1)

    sample = BatchTreeSample(tree)

    model = NarytreeLSTM(Config())
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    v = sess.run(model.get_output(), feed_dict=model.get_feed_dict(sample))
    print(v)
    return 0


def test_softmax_model():
    class Config(object):
        num_emb = 10
        emb_dim = 3
        hidden_dim = 1
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 1.0
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = True
        num_labels = 2
        embeddings = None

    tree = BatchTree.empty_tree()

    tree.root.add_sample(7, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(-1, 1, 1)
    tree.root.children[0].expand_or_add_child(3, 0, 0)
    tree.root.children[0].expand_or_add_child(3, 0, 1)
    tree.root.children[1].expand_or_add_child(3, 0, 0)
    tree.root.children[1].expand_or_add_child(3, 0, 1)

    # tree.root.add_sample(1)
    # labels = np.array([[0, 1]])
    batch_sample = BatchTreeSample(tree)

    observables, flows, mask, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, childs_transpose_scatter, nodes_count, nodes_count_per_indice = tree.build_batch_tree_sample()
    print(observables, "observables")
    print(observables_indices, "observables_indices")
    print(flows, "flows")
    print(mask, "input_scatter")
    print(scatter_out, "scatter_out")
    print(scatter_in, "scatter_in")
    print(scatter_in_indices, "scatter_in_indices")
    print(labels, "labels")
    print(out_indices, "out_indices")
    print(childs_transpose_scatter, "childs_transpose_scatter")
    print(nodes_count, "nodes_count")
    print(nodes_count_per_indice, "nodes_count_per_indice")

    labels = np.array([0, 1, 0, 1, 0])

    model = SoftMaxNarytreeLSTM(Config(), [tree])
    sess = tf.InteractiveSession()
    summarywriter = tf.summary.FileWriter('/tmp/tensortest', graph=sess.graph)
    tf.global_variables_initializer().run()
    sample = [(batch_sample, labels)]
    for i in range(100):
        model.train(batch_sample, labels, sess)
        model.test(sample, sess)
    return 0


if __name__ == '__main__':
    test_softmax_model()
    # test_lstm_model()
