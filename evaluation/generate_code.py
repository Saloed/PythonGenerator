from __future__ import print_function

import numpy as np
import tensorflow as tf

from NL2code.lang.grammar import Grammar
from current_net_conf import *
from evaluation.structures import RulesTree, NODE_VALUE_PLACEHOLDER
from model.model_single_step import RulesModelSingleStep
from model.model_single_step import WordsModelSingleStep


class CodeGenerator:
    def __init__(self, session, rules_model, words_model, rules_grammar, words_r_index, seq_end_markers,
                 counts, rules_word_placeholder):
        self.session = session  # type: tf.Session
        self.rules_model = rules_model  # type: RulesModelSingleStep
        self.words_model = words_model  # type: WordsModelSingleStep
        self.rules_grammar = rules_grammar  # type: Grammar
        self.words_r_index = words_r_index
        self.seq_end_markers = seq_end_markers
        self.counts = counts
        self.rules_word_placeholder = rules_word_placeholder

        self.rules_decoder = self.rules_model.rules_decoder
        self.words_decoder = self.words_model.words_decoder

    def rule_trees_next_step(self, rule_trees, encoder_all_states):
        decoder_input = self.rules_decoder.placeholders.inputs
        decoder_state = self.rules_decoder.placeholders.states
        query_encoder_all_states = self.rules_decoder.placeholders.query_encoder_all_states

        fetches = [
            self.rules_decoder.outputs.rules,
            self.rules_decoder.outputs.rules_decoder_new_state
        ]

        decoder_inputs = [rt.get_decoder_input() for rt in rule_trees]
        decoder_input_value = np.concatenate(decoder_inputs)

        num_layers = len(decoder_state)

        layer_states = [[] for _ in range(num_layers)]
        for rt in rule_trees:
            states = rt.get_state()
            for i in range(num_layers):
                layer_states[i].append(states[i])

        decoder_state_values = tuple([np.concatenate(st) for st in layer_states])

        many_encoder_all_states = [
            encoder_all_states
            for rt in rule_trees
        ]
        encoder_all_states_value = np.concatenate(many_encoder_all_states, axis=1)

        feed = {
            decoder_state: decoder_state_values,
            decoder_input: decoder_input_value,
            query_encoder_all_states: encoder_all_states_value
        }

        rules, new_state = self.session.run(fetches=fetches, feed_dict=feed)
        new_layer_states = [[[st] for st in states] for states in zip(*new_state)]
        return zip(rule_trees, rules, new_layer_states)

    def best_rules_sequence(self, initial_state, initial_input, encoder_all_states):
        decoder_input = self.rules_decoder.placeholders.inputs
        decoder_state = self.rules_decoder.placeholders.states
        query_encoder_all_states = self.rules_decoder.placeholders.query_encoder_all_states

        fetches = [
            self.rules_decoder.outputs.rules,
            self.rules_decoder.outputs.rules_decoder_new_state
        ]

        feed = {
            decoder_state: initial_state,
            decoder_input: initial_input,
            query_encoder_all_states: encoder_all_states
        }

        end_marker = self.seq_end_markers['rules']
        rules_count = self.counts['rules']
        rule_sequence = []
        rule_id_sequence = []

        for t in range(100):
            rules, new_state = self.session.run(fetches=fetches, feed_dict=feed)
            best_rule = np.argmax(rules)

            if best_rule == end_marker:
                break

            next_input = [np.eye(rules_count)[best_rule]]

            feed = {
                decoder_state: new_state,
                decoder_input: next_input,
                query_encoder_all_states: encoder_all_states
            }
            _best_rule = self.rules_grammar.id_to_rule[best_rule]
            rule_id_sequence.append(best_rule)
            rule_sequence.append(_best_rule)
        return rule_sequence

    def beam_search_rules(self, encoder_last_state, encoder_all_states):
        decoder_initial_input = self.rules_decoder.initializer['rules_decoder_inputs']()
        decoder_initial_state = self.rules_decoder.initializer['rules_decoder_state'](encoder_last_state)

        # best_rule_sequence = best_rules_sequence(session, rules_decoder, rules_grammar, decoder_initial_state,
        #                                      decoder_initial_input, encoder_all_states, rules_end_marker, rules_count)
        rules_count = self.counts['rules']
        rules_tree_root = RulesTree.create_new(self.rules_grammar)
        rules_tree_root.set_state(decoder_initial_state)
        rules_tree_root.set_decoder_input(decoder_initial_input)
        rules_tree_root.initialize_decoder_input_source(rules_count)

        active_rule_trees = [rules_tree_root]
        completed_rule_trees = []

        for t in range(100):
            rule_trees = self.rule_trees_next_step(active_rule_trees, encoder_all_states)

            new_rule_trees = []
            for rule_tree, next_rule_probs, next_state in rule_trees:
                frontier_nt = rule_tree.frontier_nt()
                possible_rules = self.rules_grammar[frontier_nt.as_type_node]
                for rule in possible_rules:
                    rule_id = self.rules_grammar.rule_to_id[rule]
                    score = -np.log(next_rule_probs[rule_id])
                    # score = next_rule_probs[rule_id]
                    new_tree = rule_tree.apply(rule, score)
                    new_tree.set_state(next_state)
                    new_rule_trees.append(new_tree)

            not_completed_rule_trees = []
            for rule_tree in new_rule_trees:
                if rule_tree.is_finished():
                    completed_rule_trees.append(rule_tree)
                else:
                    not_completed_rule_trees.append(rule_tree)

            if len(completed_rule_trees) > RULES_BEAM_SIZE:
                break

            not_completed_rule_trees.sort(key=lambda rt: rt.score)

            beam_size = 0
            # best_score = not_completed_rule_trees[0].score
            # for rt in not_completed_rule_trees:
            #     if abs(best_score - rt.score) > RULES_BEAM_SCORE_DELTA:
            #         break
            #     beam_size += 1

            beam_size = max(RULES_BEAM_SIZE, beam_size)
            active_rule_trees = not_completed_rule_trees[:beam_size]

        completed_rule_trees.sort(key=lambda rt: rt.score)
        return completed_rule_trees[:RULES_BEAM_SIZE]

    def words_next_step(self, query_states, state, inputs, raw_query):
        decoder_input = self.words_decoder.placeholders.words_decoder_inputs
        decoder_state = self.words_decoder.placeholders.words_decoder_state
        query_encoder_states = self.words_decoder.placeholders.query_encoder_states

        decoder_fetches = [
            self.words_decoder.outputs.words_decoder_new_state,
            self.words_decoder.outputs.words_logits,
            self.words_decoder.outputs.copy_scores,
            self.words_decoder.outputs.generated_tokens,
            self.words_decoder.outputs.generate_or_copy_score
        ]

        decoder_feed = {dst: dstv for dst, dstv in zip(decoder_state, state)}

        decoder_feed.update({
            decoder_input: inputs,
            query_encoder_states: query_states
        })

        next_state, next_input, copy_score, gen_token, gen_or_copy = self.session.run(
            fetches=decoder_fetches,
            feed_dict=decoder_feed
        )

        copy_prob, gen_prob = gen_or_copy[0][0]

        if gen_prob > copy_prob:
            gen_id = np.argmax(gen_token[0][0])
            result = self.words_r_index[gen_id]

        if copy_prob >= gen_prob or result == '<unk>':
            copy_id = np.argmax(copy_score[0][0])
            result = raw_query[copy_id]

        return result, next_state, next_input

    def generate_words(self, words_encoder_last_state, query_encoder_all_states, raw_query):
        words_end_marker = self.seq_end_markers['words']
        end_marker = self.words_r_index[words_end_marker]
        decoder_initial_input = self.words_decoder.initializer['words_decoder_inputs']()
        decoder_initial_state = self.words_decoder.initializer['words_decoder_state'](words_encoder_last_state)

        result_words = []

        next_state, next_input = decoder_initial_state, decoder_initial_input

        for t in range(100):
            word, next_state, next_input = self.words_next_step(
                query_states=query_encoder_all_states,
                state=next_state,
                inputs=next_input,
                raw_query=raw_query
            )

            if word == end_marker:
                break

            result_words.append(word)

        return result_words

    def join_word_groups(self, word_groups):
        return [
            ''.join(word_group)
            for word_group in word_groups
        ]

    def split_words_by_eos(self, words):
        result = []
        word_group = []
        for word in words:
            if word == '<eos>':
                if word_group:
                    result.append(word_group)
                word_group = []
            else:
                word_group.append(word)
        if word_group:
            result.append(word_group)

        return self.join_word_groups(result)

    def fill_placeholders(self, rule_tree, words):

        class Context:
            word_id = 0

        def traversal(node):
            if node.value is not None and node.value == NODE_VALUE_PLACEHOLDER:
                if Context.word_id < len(words):
                    node.value = words[Context.word_id]
                    Context.word_id += 1
                else:
                    node.value = '__PLACEHOLDER__'
                return
            if node.children:
                for child in node.children:
                    traversal(child)

        decode_tree = rule_tree.tree
        traversal(decode_tree)
        return decode_tree

    def get_rule_sequence(self, rules):
        rules_with_pc = []
        for rule in rules:
            rule_id = self.rules_grammar.rule_to_id[rule]
            rules_with_pc.append(rule_id)
            value_nodes = [l for l in rule.get_leaves() if self.rules_grammar.is_value_node(l.as_type_node)]
            for vnode in value_nodes:
                rules_with_pc.append(self.rules_word_placeholder)
        return rules_with_pc

    def generate_words_for_rule_tree(self, rule_tree, query, raw_query):
        rule_id_sequence = self.get_rule_sequence(rule_tree.rules)
        query_encoder_pc = self.words_model.query_encoder.placeholders
        query_encoder_feed = {
            query_encoder_pc.query_ids: [[token] for token in query],
            query_encoder_pc.query_length: [len(query)],
        }
        query_encoder_all_states = self.session.run(
            fetches=self.words_model.query_encoder.outputs.all_states,
            feed_dict=query_encoder_feed
        )

        words_encoder_pc = self.words_model.words_encoder.placeholders
        words_encoder_feed = {
            words_encoder_pc.words_rules_seq: [[rule_id] for rule_id in rule_id_sequence],
            words_encoder_pc.words_rules_seq_len: [len(rule_id_sequence)]
        }
        words_encoder_last_state = self.session.run(
            fetches=self.words_model.words_encoder.outputs.last_state,
            feed_dict=words_encoder_feed
        )

        words = self.generate_words(
            words_encoder_last_state=words_encoder_last_state,
            query_encoder_all_states=query_encoder_all_states,
            raw_query=raw_query
        )

        words_for_placement = self.split_words_by_eos(words)

        return rule_tree, words_for_placement

    def generate_code_for_query(self, query, raw_query):
        query_encoder_pc = self.rules_model.query_encoder.placeholders
        query_encoder_feed = {
            query_encoder_pc.query_ids: [[token] for token in query],
            query_encoder_pc.query_length: [len(query)],
        }
        query_encoder_fetches = [
            self.rules_model.query_encoder.outputs.last_state,
            self.rules_model.query_encoder.outputs.all_states
        ]
        query_encoder_last_state, query_encoder_all_states = self.session.run(
            fetches=query_encoder_fetches,
            feed_dict=query_encoder_feed
        )

        rule_trees = self.beam_search_rules(
            encoder_last_state=query_encoder_last_state,
            encoder_all_states=query_encoder_all_states,
        )

        rule_trees_with_words = [
            self.generate_words_for_rule_tree(
                rule_tree=rt,
                query=query,
                raw_query=raw_query
            )
            for rt in rule_trees
        ]
        result = self.select_best_tree(rule_trees_with_words)
        return result

    def select_best_tree(self, rule_trees_with_words):
        return 3
