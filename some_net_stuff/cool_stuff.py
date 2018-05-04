import os
import struct

from .batch_tree import BatchTree, BatchTreeSample


class tNode(object):
    def __init__(self, idx=-1, word=None):
        self.left = None
        self.right = None
        self.word = word
        self.size = 0
        self.leaf_pos = None
        self.height = 1
        self.parent = None
        self.label = None
        self.children = []
        self.idx = idx
        self.span = None

    def add_parent(self, parent):
        self.parent = parent

    def add_child(self, node):
        assert len(self.children) < 2
        self.children.append(node)

    def add_children(self, children):
        self.children.extend(children)

    def get_left(self):
        left = None
        if self.children:
            left = self.children[0]
        return left

    def get_right(self):
        right = None
        if len(self.children) == 2:
            right = self.children[1]
        return right

    @staticmethod
    def get_height(root):
        if root.children:
            root.height = max(root.get_left().height, root.get_right().height) + 1
        else:
            root.height = 1
        print(root.idx, root.height, 'asa')

    @staticmethod
    def get_size(root):
        if root.children:
            root.size = root.get_left().size + root.get_right().size + 1
        else:
            root.size = 1

    @staticmethod
    def get_spans(root):
        if root.children:
            root.span = root.get_left().span + root.get_right().span
        else:
            root.span = [root.word]

    @staticmethod
    def get_numleaves(self):
        if self.children:
            self.num_leaves = self.get_left().num_leaves + self.get_right().num_leaves
        else:
            self.num_leaves = 1

    @staticmethod
    def postOrder(root, func=None, args=None):

        if root is None:
            return
        tNode.postOrder(root.get_left(), func, args)
        tNode.postOrder(root.get_right(), func, args)

        if args is not None:
            func(root, args)
        else:
            func(root)

    @staticmethod
    def encodetokens(root, func):
        if root is None:
            return
        if root.word is None:
            return
        else:
            root.word = func(root.word)

    @staticmethod
    def relabel(root, fine_grained):
        if root is None:
            return
        if root.label is not None:
            if fine_grained:
                root.label += 2
            else:
                if root.label < 0:
                    root.label = 0
                elif root.label == 0:
                    root.label = 1
                else:
                    root.label = 2

    @staticmethod
    def compute_leaf_pos(root, prev_leaf_pos=-1):
        if root is None:
            return prev_leaf_pos
        if root.children:
            prev_leaf_pos = tNode.compute_leaf_pos(root.get_left(), prev_leaf_pos)
            return tNode.compute_leaf_pos(root.get_right(), prev_leaf_pos)
        else:
            root.leaf_pos = prev_leaf_pos + 1
            return prev_leaf_pos

    @staticmethod
    def compute_labels(root, labels):
        if root is None:
            return labels
        if root.children:
            labels = tNode.compute_leaf_pos(root.get_right(), labels)
            labels = tNode.compute_leaf_pos(root.get_left(), labels)
            root.label = max(root.get_right().label, root.get_left().label)
            return labels
        else:
            assert len(labels) > 0
            root.label = labels.pop()
            return labels


def test_tNode():
    nodes = {}
    for i in range(7):
        nodes[i] = tNode(i)
        if i < 4: nodes[i].word = i + 10
    nodes[0].parent = nodes[1].parent = nodes[4]
    nodes[2].parent = nodes[3].parent = nodes[5]
    nodes[5].parent = nodes[6].parent = nodes[6]
    nodes[6].add_child(nodes[4])
    nodes[6].add_child(nodes[5])
    nodes[4].add_children([nodes[0], nodes[1]])
    nodes[5].add_children([nodes[2], nodes[3]])
    root = nodes[6]
    postOrder = root.postOrder
    postOrder(root, tNode.get_height, None)
    postOrder(root, tNode.get_numleaves, None)
    postOrder(root, root.get_spans, None)
    print(root.height, root.num_leaves)
    for n in nodes.values():
        print(n.span)


def processTree(root: tNode, funclist=None, argslist=None):
    if funclist is None:
        root.postOrder(root, root.get_height)
        root.postOrder(root, root.get_num_leaves)
        root.postOrder(root, root.get_size)
    else:
        # print funclist,argslist
        for func, args in zip(funclist, argslist):
            root.postOrder(root, func, args)

    return root


if __name__ == '__main__':
    test_tNode()


class Vocab(object):

    def __init__(self, path):
        self.words = set()
        self.word2idx = {}
        self.idx2word = {}
        self.embed_matrix = None
        self.load(path)

    def load(self, path):

        with open(path, 'r') as f:
            for line in f:
                w = line.strip()
                assert w not in self.words
                self.words.add(w)
                self.word2idx[w] = len(self.words) - 1  # 0 based index
                self.idx2word[self.word2idx[w]] = w

    def __len__(self):
        return len(self.words)

    def encode(self, word):
        if word not in self.words:
            return None
        return self.word2idx[word]

    def decode(self, idx):
        assert idx < len(self.words) and idx >= 0
        return self.idx2word[idx]

    def size(self):
        return len(self.words)

    def compute_gloves_embedding(self, glove_dir):
        vector_format = 'f' * 300
        size = struct.calcsize(vector_format)
        glove_voc = Vocab(os.path.join(glove_dir, 'glove_vocab.txt'))
        self.embed_matrix = np.random.rand(self.size(), 300) * 0.1 - 0.05
        with open(os.path.join(glove_dir, 'glove_embeddings.b'), "rb") as fi:
            id = 0
            while True:
                vector = fi.read(size)
                if len(vector) == size:
                    v = struct.unpack(vector_format, vector)
                    idx = self.encode(glove_voc.decode(id))
                    if idx is not None:
                        self.embed_matrix[idx, :] = v
                else:
                    break
                id += 1
                if id % 100000 == 0:
                    print("read " + str(id) + " embeds and counting...")


def load_sentiment_treebank(data_dir, glove_dir, fine_grained):
    voc = Vocab(os.path.join(data_dir, 'vocab-cased.txt'))
    if glove_dir is not None:
        voc.compute_gloves_embedding(glove_dir)

    split_paths = {}
    for split in ['train', 'test', 'dev']:
        split_paths[split] = os.path.join(data_dir, split)

    fnlist = [tNode.encodetokens, tNode.relabel]
    arglist = [voc.encode, fine_grained]
    # fnlist,arglist=[tNode.relabel],[fine_grained]

    data = {}
    for split, path in split_paths.iteritems():
        sentencepath = os.path.join(path, 'sents.txt')
        treepath = os.path.join(path, 'parents.txt')
        labelpath = os.path.join(path, 'labels.txt')
        trees = parse_trees(sentencepath, treepath, labelpath)
        if not fine_grained:
            trees = [tree for tree in trees if tree.label != 0]
        trees = [(processTree(tree, fnlist, arglist), tree.label) for tree in trees]
        data[split] = trees

    return data, voc


def load_subtitles(data_dir, only_supervised_data=False):
    voc = Vocab(os.path.join(data_dir, 'vocab.txt'))

    fnlist = [tNode.encodetokens]
    arglist = [voc.encode]
    # fnlist,arglist=[tNode.relabel],[fine_grained]

    sentencepath = os.path.join(data_dir, 'subtitles.txt')
    treepath = os.path.join(data_dir, 'subtitles.cparents')
    labelpath = os.path.join(data_dir, 'labels.txt')
    trees = parse_trees(sentencepath, treepath, labelpath)
    with open(labelpath) as fl:
        label_line = fl.readline()
        labels = [-1 if (l[1] == "?" or l[1] == "#") else (1 if l[1] == "1" else 0) for l in label_line.split()]
        trees = [(processTree(tree, fnlist + tNode.compute_labels, arglist + [labels]), tree.label) for tree in trees]
        if only_supervised_data:
            trees = [tree for tree in trees if tree.label >= 0]
        data = trees
    return data, voc


def parse_trees(sentencepath, treepath, labelpath):
    trees = []
    with open(treepath, 'r') as ft, open(
            sentencepath, 'r') as f:
        fl = None
        if labelpath is not None:
            fl = open(labelpath)

        while True:
            parentidxs = ft.readline()
            labels = None
            if fl is not None:
                labels = fl.readline()
                labels = [int(l) if l != '#' and l != '?' else None for l in labels.strip().split()]

            sentence = f.readline()
            if not parentidxs or not labels or not sentence:
                break
            parentidxs = [int(p) for p in parentidxs.strip().split()]

            tree = parse_tree(sentence, parentidxs, labels)
            trees.append(tree)
    if fl is not None:
        fl.close()

    return trees


def parse_tree(sentence, parents, labels):
    nodes = {}
    parents = [p - 1 for p in parents]  # change to zero based
    sentence = [w for w in sentence.strip().split()]
    for i in range(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx)
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)

                if labels is not None:
                    node.label = labels[idx]
                else:
                    node.label = None
                nodes[idx] = node

                if idx < len(sentence):
                    node.word = sentence[idx]

                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    return root


def BFStree(root):
    from collections import deque
    node = root
    leaves = []
    inodes = []
    queue = deque([node])
    func = lambda node: node.children == []

    while queue:
        node = queue.popleft()
        if func(node):
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            queue.extend(node.children)

    return leaves, inodes


def extract_tree_data(tree, max_degree=2, only_leaves_have_vals=True, with_labels=False):
    # processTree(tree)
    # fnlist=[tree.encodetokens,tree.relabel]
    # arglist=[voc.encode,fine_grained]
    # processTree(tree,fnlist,arglist)
    leaves, inodes = BFStree(tree)
    labels = []
    leaf_emb = []
    tree_str = []
    i = 0
    for leaf in reversed(leaves):
        leaf.idx = i
        i += 1
        labels.append(leaf.label)
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx = i
        c = [child.idx for child in node.children]
        tree_str.append(c)
        labels.append(node.label)
        if not only_leaves_have_vals:
            leaf_emb.append(-1)
        i += 1
    if with_labels:
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(leaf_emb, dtype='int32'),
                np.array(tree_str, dtype='int32'),
                np.array(labels, dtype=float),
                np.array(labels_exist, dtype=float))
    else:
        print(leaf_emb, 'asas')
        return (np.array(leaf_emb, dtype='int32'),
                np.array(tree_str, dtype='int32'))


def build_batch_trees(trees, mini_batch_size):
    def expand_batch_with_sample(batch_node, sample_node):
        if batch_node.parent is None:  # root
            batch_node.add_sample(-1 if sample_node.word is None else sample_node.word, tree.label)
        for child in zip(range(len(sample_node.children)), sample_node.children):
            batch_node.expand_or_add_child(child[1].word, child[1].label, child[0])
        for children in zip(batch_node.children, sample_node.children):
            expand_batch_with_sample(children[0], children[1])

    batches = []
    while len(trees) > 0:
        batch = trees[-mini_batch_size:]
        del trees[-mini_batch_size:]
        batch_tree = BatchTree.empty_tree()
        for tree in batch:
            expand_batch_with_sample(batch_tree.root, tree)
        batches.append(BatchTreeSample(batch_tree))
    return batches


def build_labelized_batch_trees(data, mini_batch_size):
    trees = [s[0] for s in data]
    labels = [s[1] for s in data]
    labels_batches = []
    while len(labels) > 0:
        batch = labels[-mini_batch_size:]
        del labels[-mini_batch_size:]
        labels_batches.append(np.array(batch))
    tree_batches = build_batch_trees(trees, mini_batch_size)
    return zip(tree_batches, labels_batches)


def extract_batch_tree_data(batchdata, fillnum=120):
    dim1, dim2 = len(batchdata), fillnum
    # leaf_emb_arr,treestr_arr,labels_arr=[],[],[]
    leaf_emb_arr = np.empty([dim1, dim2], dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim1, dim2, 2], dtype='int32')
    treestr_arr.fill(-1)
    labels_arr = np.empty([dim1, dim2], dtype=float)
    labels_arr.fill(-1)
    for i, (tree, _) in enumerate(batchdata):
        input_, treestr, labels, _ = extract_tree_data(tree,
                                                       max_degree=2,
                                                       only_leaves_have_vals=False,
                                                       with_labels=True)
        leaf_emb_arr[i, 0:len(input_)] = input_
        treestr_arr[i, 0:len(treestr), 0:2] = treestr
        labels_arr[i, 0:len(labels)] = labels

    return leaf_emb_arr, treestr_arr, labels_arr


def extract_seq_data(data, numsamples=0, fillnum=100):
    seqdata = []
    seqlabels = []
    for tree, _ in data:
        seq, seqlbls = extract_seq_from_tree(tree, numsamples)
        seqdata.extend(seq)
        seqlabels.extend(seqlbls)

    seqlngths = [len(s) for s in seqdata]
    maxl = max(seqlngths)
    assert fillnum >= maxl
    if 1:
        seqarr = np.empty([len(seqdata), fillnum], dtype='int32')
        seqarr.fill(-1)
        for i, s in enumerate(seqdata):
            seqarr[i, 0:len(s)] = np.array(s, dtype='int32')
        seqdata = seqarr
    return seqdata, seqlabels, seqlngths, maxl


def extract_seq_from_tree(tree, numsamples=0):
    if tree.span is None:
        tree.postOrder(tree, tree.get_spans)

    seq, lbl = [], []
    s, l = tree.span, tree.label
    seq.append(s)
    lbl.append(l)

    if not numsamples:
        return seq, lbl

    num_nodes = tree.idx
    if numsamples == -1:
        numsamples = num_nodes
    # numsamples=min(numsamples,num_nodes)
    # sampled_idxs = random.sample(range(num_nodes),numsamples)
    # sampled_idxs=range(num_nodes)
    # print sampled_idxs,num_nodes

    subtrees = {}

    # subtrees[tree.idx]=
    # func=lambda tr,su:su.update([(tr.idx,tr)])
    def func_(self, su):
        su.update([(self.idx, self)])

    tree.postOrder(tree, func_, subtrees)

    for j in range(numsamples):  # sampled_idxs:
        i = random.randint(0, num_nodes)
        root = subtrees[i]
        s, l = root.span, root.label
        seq.append(s)
        lbl.append(l)

    return seq, lbl


def get_max_len_data(datadic):
    maxlen = 0
    for data in datadic.values():
        for tree, _ in data:
            tree.postOrder(tree, tree.get_numleaves)
            assert tree.num_leaves > 1
            if tree.num_leaves > maxlen:
                maxlen = tree.num_leaves

    return maxlen


def get_max_node_size(datadic):
    maxsize = 0
    for data in datadic.values():
        for tree, _ in data:
            tree.postOrder(tree, tree.get_size)
            assert tree.size > 1
            if tree.size > maxsize:
                maxsize = tree.size

    return maxsize


def test_fn():
    data_dir = './stanford_lstm/data/sst'
    fine_grained = 0
    data, _ = load_sentiment_treebank(data_dir, fine_grained)
    for d in data.itervalues():
        print(len(d))

    d = data['dev']
    a, b, c, _ = extract_seq_data(d[0:1], 5)
    print(a, b, c)

    print(get_max_len_data(data))
    return data


if __name__ == '__main__':
    test_fn()

import sys
import numpy as np
import tensorflow as tf
import random

from . import TreeLSTM as nary_tree_lstm

DIR = 'data/sst/'
GLOVE_DIR = 'data/glove/'

import time


# from tf_data_utils import extract_tree_data,load_sentiment_treebank

class Config(object):
    num_emb = None

    emb_dim = 300
    hidden_dim = 150
    output_dim = None
    degree = 2
    num_labels = 3
    num_epochs = 50

    maxseqlen = None
    maxnodesize = None
    fine_grained = False
    trainable_embeddings = True
    nonroot_labels = True

    embeddings = None


def train2():
    config = Config()
    config.batch_size = 25
    config.lr = 0.05
    config.dropout = 0.5
    config.reg = 0.0001
    config.emb_lr = 0.02

    import collections
    import numpy as np
    from sklearn import metrics

    def test(model, data, session):
        relevant_labels = [0, 2]
        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = model.get_output()
            y_true = batch[0].root_labels / 2
            feed_dict = {model.labels: batch[0].root_labels}
            feed_dict.update(model.tree_lstm.get_feed_dict(batch[0]))
            y_pred_ = session.run([y_pred], feed_dict=feed_dict)
            y_pred_ = np.argmax(y_pred_[0][:, relevant_labels], axis=1)
            ys_true += y_true.tolist()
            ys_pred += y_pred_.tolist()
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print("Accuracy", score)
        # print "Recall", metrics.recall_score(ys_true, ys_pred)
        # print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print("confusion_matrix")
        print(metrics.confusion_matrix(ys_true, ys_pred))
        return score

    data, vocab = load_sentiment_treebank(DIR, GLOVE_DIR, config.fine_grained)
    # data, vocab = load_sentiment_treebank(DIR, None, config.fine_grained)
    config.embeddings = vocab.embed_matrix

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print('train', len(train_set))
    print('dev', len(dev_set))
    print('test', len(test_set))

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(range(num_labels)), set(labels)
    print('num emb', num_emb)
    print('num labels', num_labels)

    config.num_emb = num_emb
    config.output_dim = num_labels

    # return
    random.seed()
    np.random.seed()

    from random import shuffle
    shuffle(train_set)
    train_set = build_labelized_batch_trees(train_set, config.batch_size)
    dev_set = build_labelized_batch_trees(dev_set, 500)
    test_set = build_labelized_batch_trees(test_set, 500)

    with tf.Graph().as_default():

        # model = tf_seq_lstm.tf_seqLSTM(config)
        model = nary_tree_lstm.SoftMaxNarytreeLSTM(config, train_set + dev_set + test_set)

        init = tf.global_variables_initializer()
        best_valid_score = 0.0
        best_valid_epoch = 0
        dev_score = 0.0
        test_score = 0.0
        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(config.num_epochs):
                start_time = time.time()
                print('epoch', epoch)
                avg_loss = 0.0
                model.train_epoch(train_set[:], sess)

                print("Training time per epoch is {0}".format(
                    time.time() - start_time))

                print('validation score')
                score = test(model, dev_set, sess)
                # print 'train score'
                # test(model, train_set[:40], sess)
                if score >= best_valid_score:
                    best_valid_score = score
                    best_valid_epoch = epoch
                    test_score = test(model, test_set, sess)
                print('test score :', test_score, 'updated', epoch - best_valid_epoch,
                      'epochs ago with validation score', best_valid_score)


def train(restore=False):
    config = Config()
    config.batch_size = 5
    config.lr = 0.05
    data, vocab = load_sentiment_treebank(DIR, GLOVE_DIR, config.fine_grained)
    config.embeddings = vocab.embed_matrix
    config.early_stopping = 2
    config.reg = 0.0001
    config.dropout = 1.0
    config.emb_lr = 0.1

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print('train', len(train_set))
    print('dev', len(dev_set))
    print('test', len(test_set))

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(range(num_labels)), set(labels)
    print('num emb', num_emb)
    print('num labels', num_labels)

    config.num_emb = num_emb
    config.output_dim = num_labels

    config.maxseqlen = get_max_len_data(data)
    config.maxnodesize = get_max_node_size(data)

    print(config.maxnodesize, config.maxseqlen, " maxsize")
    # return
    random.seed()
    np.random.seed()

    with tf.Graph().as_default():

        # model = tf_seq_lstm.tf_seqLSTM(config)
        model = nary_tree_lstm.tf_NarytreeLSTM(config)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        best_valid_score = 0.0
        best_valid_epoch = 0
        dev_score = 0.0
        test_score = 0.0
        with tf.Session() as sess:

            sess.run(init)

            if restore: saver.restore(sess, './ckpt/tree_rnn_weights')
            for epoch in range(config.num_epochs):
                start_time = time.time()
                print('epoch', epoch)
                avg_loss = 0.0
                avg_loss = train_epoch(model, train_set, sess)
                print('avg loss', avg_loss)

                print("Training time per epoch is {0}".format(
                    time.time() - start_time))

                dev_score = evaluate(model, dev_set, sess)
                print('dev-score', dev_score)

                if dev_score >= best_valid_score:
                    best_valid_score = dev_score
                    best_valid_epoch = epoch
                    # saver.save(sess,'./ckpt/tree_rnn_weights')
                    test_score = evaluate(model, test_set, sess)
                    print('test score :', test_score, 'updated', epoch - best_valid_epoch,
                          'epochs ago with validation score', best_valid_score)


def train_epoch(model, data, sess):
    loss = model.train(data, sess)
    return loss


def evaluate(model, data, sess):
    acc = model.evaluate(data, sess)
    return acc


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "-optimized":
            print("running optimized version")
            train2()
        else:
            print("running not optimized version")
            train()
    else:
        print("running not optimized version, run with option -optimized for the optimized one")
        train()
