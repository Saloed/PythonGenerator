import os
import re
import time

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from analyser import Embeddings
from analyser.Options import Options
from analyser.Score import Score
from analyser.analyser_rnn import sequence_input, labels_output, tree_tokens_output, strings_output, \
    sequence_tokens_output
from analyser.attention_dynamic_rnn import Attention
from analyser.misc import cross_entropy_loss, l2_loss, batch_greedy_correct, calc_scores, print_diff, \
    print_scores, newest
from contants import PAD, NOP, UNDEFINED
from logger import logger
from prepares import DataSet
from utils.Formatter import Formatter
from utils.SummaryWriter import SummaryWriter
from utils.wrappers import trace, Timer


class AnalyserNet:
    @trace("BUILD NET")
    def __init__(self, options: Options, data_set: DataSet, dtype=None, scope=None):
        self.options = options
        self.options.validate()
        num_labels = len(Embeddings.labels())
        num_tokens = len(Embeddings.tokens())
        num_words = len(Embeddings.words())
        undefined = Embeddings.labels().get_index(UNDEFINED)
        nop = Embeddings.tokens().get_index(NOP)
        pad = Embeddings.words().get_index(PAD)
        with tf.variable_scope("Input"), Timer("BUILD INPUT"):
            self.inputs = tf.placeholder(tf.int32, [self.options.batch_size, None], "inputs")
            self.inputs_length = tf.placeholder(tf.int32, [self.options.batch_size], "inputs_length")
            self.labels_length = tf.placeholder(tf.int32, [], "labels_length")
            self.tokens_length = tf.placeholder(tf.int32, [], "tokens_length")
            self.strings_length = tf.placeholder(tf.int32, [], "strings_length")
            inputs = tf.gather(tf.constant(np.asarray(Embeddings.words().idx2emb)), self.inputs)
        with tf.variable_scope(scope or "Analyser", dtype=dtype) as scope, Timer("BUILD BODY"):
            dtype = scope.dtype
            cell_fw = GRUCell(self.options.inputs_state_size)
            cell_bw = GRUCell(self.options.inputs_state_size)
            attention_states = sequence_input(
                cell_fw, cell_bw, inputs, self.inputs_length, self.options.inputs_hidden_size, dtype)
            labels_attention = Attention(
                attention_states, self.options.labels_state_size, dtype=dtype, scope="LabelsAttention")
            labels_cell = GRUCell(self.options.labels_state_size)
            self.labels_logits, self.raw_labels, labels_states, attentions, weights = labels_output(
                labels_cell, labels_attention, num_labels, self.labels_length,
                hidden_size=self.options.labels_hidden_size, dtype=dtype)
            tokens_attention = Attention(
                attention_states, self.options.tokens_state_size, dtype=dtype, scope="TokensAttention")
            if options.tokens_output_type == "tree":
                tokens_left_cell = GRUCell(self.options.tokens_state_size)
                tokens_right_cell = GRUCell(self.options.tokens_state_size)
                tokens_cell = (tokens_left_cell, tokens_right_cell)
                self.tokens_logits, self.raw_tokens, tokens_states, attentions, weights = tree_tokens_output(
                    tokens_cell, tokens_attention, num_tokens, self.tokens_length, labels_states,
                    hidden_size=self.options.tokens_hidden_size, dtype=dtype)
            elif options.tokens_output_type == "sequence":
                tokens_cell = GRUCell(self.options.tokens_state_size)
                self.tokens_logits, self.raw_tokens, tokens_states, attentions, weights = sequence_tokens_output(
                    tokens_cell, tokens_attention, num_tokens, self.tokens_length, labels_states,
                    hidden_size=self.options.tokens_hidden_size, dtype=dtype)
            else:
                raise ValueError("Tokens output type '%s' hasn't recognised" % options.tokens_output_type)
            strings_attention = Attention(
                attention_states, self.options.strings_state_size, dtype=dtype, scope="StringsAttention")
            strings_cell = GRUCell(self.options.strings_state_size)
            self.strings_logits, self.raw_strings, strings_states, attentions, weights = strings_output(
                strings_cell, strings_attention, num_words, self.strings_length, tokens_states,
                hidden_size=self.options.strings_hidden_size, dtype=dtype)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        with tf.variable_scope("Output"), Timer("BUILD OUTPUT"):
            def confident(raw_data, axis, default, confidence):
                probs = tf.reduce_max(raw_data, axis)
                mask = tf.to_int64(tf.greater(probs, confidence))
                data = tf.argmax(raw_data, axis)
                return mask * data + (1 - mask) * default

            self.labels = confident(self.raw_labels, 2, undefined, self.options.label_confidence)
            self.tokens = confident(self.raw_tokens, 3, nop, self.options.token_confidence)
            self.strings = confident(self.raw_strings, 4, pad, self.options.string_confidence)
        with tf.variable_scope("Loss"), Timer("BUILD LOSS"):
            self.labels_targets = tf.placeholder(tf.int32, [self.options.batch_size, None], "labels")
            self.tokens_targets = tf.placeholder(tf.int32, [self.options.batch_size, None, None], "tokens")
            self.strings_targets = tf.placeholder(tf.int32, [self.options.batch_size, None, None, None], "strings")
            self.labels_loss = cross_entropy_loss(self.labels_targets, self.labels_logits, undefined)
            self.tokens_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits, nop)
            self.strings_loss = cross_entropy_loss(self.strings_targets, self.strings_logits, pad)
            self.l2_loss = self.options.l2_weight * l2_loss(self.variables)
            self.loss = self.labels_loss + self.tokens_loss + self.strings_loss + self.l2_loss
        with tf.variable_scope("Optimizer"), Timer("BUILD OPTIMISER"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        with tf.variable_scope("Summaries"), Timer("BUILD SUMMARIES"):
            self.summaries = self.add_variable_summaries()
        self.saver = tf.train.Saver(var_list=self.variables)
        self.save_path = self.options.model_dir
        self.data_set = data_set

    def save(self, session: tf.Session):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        save_time = time.strftime("%d-%m-%Y-%H-%M-%S")
        model_path = os.path.join(self.save_path, "model-%s.ckpt" % save_time)
        self.saver.save(session, model_path)

    def restore(self, session: tf.Session):
        save_path = self.save_path
        model_pattern = re.compile(r"model-\d{1,2}-\d{1,2}-\d{4}-\d{1,2}-\d{1,2}-\d{1,2}\.ckpt\.meta")
        filtrator = lambda path, name: os.path.isfile(path + "/" + name) and re.match(model_pattern, name)
        model_path = newest(save_path, filtrator)
        model_path = ".".join(model_path.split(".")[:-1])
        self.saver.restore(session, model_path)

    def add_variable_summaries(self):
        variables = [tf.reshape(variable, [-1]) for variable in self.variables]
        tf.summary.histogram("Summary", tf.concat(variables, 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name.replace(":", "_"), variable)
        return tf.summary.merge_all()

    def build_feed_dict(self, batch) -> dict:
        (inputs, inputs_length), labels, tokens, strings = batch
        labels_targets, labels_length = labels
        tokens_targets, tokens_length = tokens
        strings_targets, strings_length = strings
        feed_dict = {
            self.inputs: inputs,
            self.inputs_length: inputs_length,
            self.labels_length: labels_length,
            self.tokens_length: tokens_length,
            self.strings_length: strings_length,
            self.labels_targets: labels_targets,
            self.tokens_targets: tokens_targets,
            self.strings_targets: strings_targets}
        return feed_dict

    def correct_target(self, feed_dict, session) -> dict:
        fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
        outputs = session.run(fetches, feed_dict)
        labels_targets = feed_dict[self.labels_targets]
        tokens_targets = feed_dict[self.tokens_targets]
        strings_targets = feed_dict[self.strings_targets]
        undefined = Embeddings.labels().get_index(UNDEFINED)
        nop = Embeddings.tokens().get_index(NOP)
        pad = Embeddings.words().get_index(PAD)
        _labels_targets = np.copy(labels_targets)
        _tokens_targets = np.copy(tokens_targets)
        _strings_targets = np.copy(strings_targets)
        _labels_targets[_labels_targets == -1] = undefined
        _tokens_targets[_tokens_targets == -1] = nop
        _strings_targets[_strings_targets == -1] = pad
        emb_labels_targets = np.asarray(Embeddings.labels().idx2emb)[_labels_targets]
        emb_tokens_targets = np.asarray(Embeddings.tokens().idx2emb)[_tokens_targets]
        num_words = len(Embeddings.words())
        emb_strings_targets = np.eye(num_words)[_strings_targets]
        targets = (emb_labels_targets, emb_tokens_targets, emb_strings_targets)
        dependencies = (labels_targets, tokens_targets, strings_targets)
        targets, dependencies = batch_greedy_correct(targets, outputs, dependencies)
        labels_targets, tokens_targets, strings_targets = dependencies
        feed_dict[self.labels_targets] = labels_targets
        feed_dict[self.tokens_targets] = tokens_targets
        feed_dict[self.strings_targets] = strings_targets
        return feed_dict

    @trace("TRAIN")
    def train(self):
        train_loss_graphs, validation_loss_graphs = [], []
        figure0 = ProxyFigure("loss", self.save_path + "/loss.png")
        loss_labels = ("Labels", "Tokens", "Strings", "Complex")
        for i, label in enumerate(loss_labels):
            train_loss_graphs.append(figure0.smoothed_curve(1, len(loss_labels), i + 1, 0.6, mode="-b"))
            validation_loss_graphs.append(figure0.smoothed_curve(1, len(loss_labels), i + 1, 0.6, mode="-r"))
            figure0.set_x_label(1, len(loss_labels), i + 1, "epoch")
            figure0.set_label(1, len(loss_labels), i + 1, label)
        figure0.set_y_label(1, len(loss_labels), 1, "loss")

        train_score_graphs, validation_score_graphs = [], []
        figure1 = ProxyFigure("score", self.save_path + "/score.png")
        score_labels = ("Labels", "Tokens", "Strings", "Templates", "Code")
        for i, label in enumerate(score_labels):
            train_score_graphs.append(figure1.smoothed_curve(1, len(score_labels), i + 1, 0.6, mode="-b"))
            validation_score_graphs.append(figure1.smoothed_curve(1, len(score_labels), i + 1, 0.6, mode="-r"))
            figure1.set_x_label(1, len(score_labels), i + 1, "epoch")
            figure1.set_label(1, len(score_labels), i + 1, label)
        figure1.set_y_label(1, len(score_labels), 1, "score")

        loss_labels = ("l_" + label for label in loss_labels)
        score_labels = ("s_" + label for label in score_labels)
        labels = [*loss_labels, *score_labels]
        train_labels = ("t_" + label for label in labels)
        validation_labels = ("v_" + label for label in labels)
        heads = ("epoch", "time", *train_labels, *validation_labels)
        formats = ["d", ".4f"] + [".4f"] * 2 * len(labels)
        formatter = Formatter(heads, formats, [15] * len(heads), height=10)

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        session = tf.Session(config=config)
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        device = tf.device('/cpu:0')
        writer = SummaryWriter(self.options.summaries_dir, session, self.summaries, session.graph)
        with session, device, writer, figure0, figure1:
            session.run(tf.global_variables_initializer())
            for epoch in range(self.options.epochs):
                with Timer(printer=None) as timer:
                    for batch in self.data_set.train:
                        feed_dict = self.build_feed_dict(batch)
                        feed_dict = self.correct_target(feed_dict, session)
                        session.run(self.optimizer, feed_dict)
                train_losses, train_scores = self.quality(session, self.data_set.train)
                validation_losses, validation_scores = self.quality(session, self.data_set.validation)
                train_scores = [score.F_score(1) for score in train_scores]
                validation_scores = [score.F_score(1) for score in validation_scores]
                array = train_losses + train_scores + validation_losses + validation_scores
                formatter.print(epoch, timer.delay(), *array)
                train_loss_is_nan = any(np.isnan(loss) for loss in train_losses)
                validation_loss_is_nan = any(np.isnan(loss) for loss in validation_losses)
                if train_loss_is_nan or validation_loss_is_nan:
                    logger.info("NaN detected")
                    break
                self.save(session)
                for graph, value in zip(train_loss_graphs, train_losses):
                    graph.append(epoch, value)
                for graph, value in zip(validation_loss_graphs, validation_losses):
                    graph.append(epoch, value)
                for graph, value in zip(train_score_graphs, train_scores):
                    graph.append(epoch, value)
                for graph, value in zip(validation_score_graphs, validation_scores):
                    graph.append(epoch, value)
                figure0.draw()
                figure0.save()
                figure1.draw()
                figure1.save()
                writer.update()

    @trace("CROSS VALIDATION")
    def cross(self):
        def partition(train_set, validation_set, test_set):
            start = i * part_length
            data_set = train_set + validation_set
            validation_set = data_set[start: start + part_length]
            train_set = data_set[:start] + data_set[start + part_length:]
            return DataSet(train_set, validation_set, test_set)

        save_path = self.save_path
        results = []
        part_length = len(self.data_set.validation)
        num_parts = (part_length + len(self.data_set.train)) // part_length
        for i in range(num_parts):
            logger.info("Iteration #%d/%d" % (i + 1, num_parts))
            self.save_path = os.path.join(save_path, "iteration-%d" % i)
            self.data_set = partition(*self.data_set)
            self.train()
            validation = self.data_set.validation
            res = self._test(validation, False)
            results.append(res)
        return results

    def test(self):
        return self._test(self.data_set.test, True)

    @trace("TEST")
    def _test(self, test_set, show_diff):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            if show_diff:
                for batch in test_set:
                    feed_dict = self.build_feed_dict(batch)
                    feed_dict = self.correct_target(feed_dict, session)
                    labels_fetches = (self.labels_targets, self.labels)
                    tokens_fetches = (self.tokens_targets, self.tokens)
                    strings_fetches = (self.strings_targets, self.strings)
                    array = session.run(labels_fetches + tokens_fetches + strings_fetches, feed_dict)
                    inputs = feed_dict[self.inputs]
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    print_diff(inputs, *array, raw_tokens)
            losses, scores = self.quality(session, test_set)
            print_scores(scores)
        return losses, scores

    def quality(self, session, batches):
        losses, scores = [], []
        for batch in batches:
            feed_dict = self.build_feed_dict(batch)
            feed_dict = self.correct_target(feed_dict, session)
            labels_fetches = (self.labels_targets, self.labels)
            tokens_fetches = (self.tokens_targets, self.tokens)
            strings_fetches = (self.strings_targets, self.strings)
            array = session.run(labels_fetches + tokens_fetches + strings_fetches, feed_dict)
            losses_fetches = (self.labels_loss, self.tokens_loss, self.strings_loss, self.loss)
            losses.append(session.run(losses_fetches, feed_dict))
            scores.append(calc_scores(*array, self.options.flatten_type))
        losses = [np.mean(typed_losses) for typed_losses in zip(*losses)]
        scores = [Score.concat(typed_scores) for typed_scores in zip(*scores)]
        return losses, scores
