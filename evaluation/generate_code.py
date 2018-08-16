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

        many_encoder_all_states = [
            encoder_all_states
            for rt in rule_trees
        ]
        encoder_all_states_value = np.concatenate(many_encoder_all_states, axis=1)

        states_and_ctx = [rt.get_state() for rt in rule_trees]
        parent_data = [rt.get_parent_id_and_state() for rt in rule_trees]
        rules = np.stack([rt.rule_id for rt in rule_trees])
        nodes = np.stack([rt.node_id for rt in rule_trees])
        parent_rules = np.stack([rule for rule, _ in parent_data])
        attention_ctx = np.concatenate([ctx for _, ctx in states_and_ctx], axis=0)
        parent_state = np.concatenate([state[-1] for _, (state, _) in parent_data], axis=0)

        states = [state for state, _ in states_and_ctx]

        layer_states = [[] for _ in range(RULES_DECODER_LAYERS)]
        for state in states:
            for i in range(RULES_DECODER_LAYERS):
                layer_states[i].append(state[i])

        decoder_state_values = tuple([np.concatenate(st) for st in layer_states])

        feed = self.rules_decoder.placeholders.feed(
            states=decoder_state_values,
            attention_ctx=attention_ctx,
            query_states=encoder_all_states_value,
            prev_rule=rules,
            frontier_node=nodes,
            parent_rule=parent_rules,
            parent_rule_state=parent_state
        )

        fetches = self.rules_decoder.outputs.fetch_all()

        new_state, new_attention_ctx, _, rules_prob = self.session.run(fetches=fetches, feed_dict=feed)
        new_layer_states = [[[st] for st in states] for states in zip(*new_state)]
        new_attention = [[ctx] for ctx in new_attention_ctx]
        next_state = list(zip(new_layer_states, new_attention))
        return zip(rule_trees, rules_prob, next_state)

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
        decoder_initial_state = self.rules_decoder.initializer['rules_decoder_state'](encoder_last_state)
        decoder_initial_ctx = self.rules_decoder.initializer['rules_decoder_context']()

        rules_tree_root = RulesTree.create_new(self.rules_grammar)
        rules_tree_root.set_state((decoder_initial_state, decoder_initial_ctx))
        _root_parent_node_id = -1
        _root_parent_state = decoder_initial_state, None
        rules_tree_root.root_parent_data = _root_parent_node_id, _root_parent_state

        active_rule_trees = [rules_tree_root]
        completed_rule_trees = []

        for t in range(100):
            rule_trees = self.rule_trees_next_step(active_rule_trees, encoder_all_states)

            new_rule_trees = []
            for rule_tree, next_rule_probs, next_state in rule_trees:

                best_rule_ids = np.argpartition(next_rule_probs, -5)[-5:]
                best_rule_ids_sorted = best_rule_ids[np.argsort(next_rule_probs[best_rule_ids])]
                best_rules = [
                    (
                        (
                            self.rules_grammar.id_to_rule[best_rule_id]
                            if self.rules_grammar.id_to_rule.has_key(best_rule_id)
                            else best_rule_id
                        ),
                        next_rule_probs[best_rule_id]
                    )
                    for best_rule_id in reversed(best_rule_ids_sorted)
                ]

                frontier_nt = rule_tree.frontier_nt()
                possible_rules = self.rules_grammar[frontier_nt.as_type_node]
                for rule in possible_rules:
                    rule_id = self.rules_grammar.rule_to_id[rule]
                    score = np.log(next_rule_probs[rule_id])
                    # score = next_rule_probs[rule_id]
                    new_tree = rule_tree.apply(rule, score)
                    new_tree.set_state(next_state)
                    new_rule_trees.append(new_tree)

                    new_tree.best_rules = best_rules
                    new_tree.rule = rule

            not_completed_rule_trees = []
            for rule_tree in new_rule_trees:
                if rule_tree.is_finished():
                    completed_rule_trees.append(rule_tree)
                else:
                    not_completed_rule_trees.append(rule_tree)

            if len(completed_rule_trees) > RULES_BEAM_SIZE:
                break

            not_completed_rule_trees.sort(key=lambda rt: -rt.score)
            active_rule_trees = not_completed_rule_trees[:RULES_BEAM_SIZE]

        completed_rule_trees.sort(key=lambda rt: -rt.score)
        completed_rule_tree_view = [rt.view for rt in completed_rule_trees]
        return completed_rule_trees[0]

    def words_next_step(self, state, context, raw_query, encoder_states, query_states):

        decoder_feed = self.words_decoder.placeholders.decoder_feed(
            states=state,
            context=context,
            encoder_states=encoder_states
        )
        copy_feed = self.words_decoder.placeholders.copy_mechanism_feed(
            query_states=query_states
        )
        decoder_feed.update(copy_feed)

        decoder_fetches = [
            self.words_decoder.outputs.new_state,
            self.words_decoder.outputs.new_attention,
            self.words_decoder.outputs.copy_scores,
            self.words_decoder.outputs.generated_tokens,
            self.words_decoder.outputs.generate_or_copy_score
        ]

        next_state, next_context, copy_score, gen_token, gen_or_copy = self.session.run(
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

        return result, next_state, next_context

    def generate_words(self, words_encoder_last_state, words_encoder_all_states, query_encoder_all_states, raw_query):

        words_end_marker = self.seq_end_markers['words']
        end_marker = self.words_r_index[words_end_marker]

        decoder_initial_context = self.words_decoder.initializer['words_decoder_context']()
        decoder_initial_state = self.words_decoder.initializer['words_decoder_state'](words_encoder_last_state)

        result_words = []

        next_state, next_context = decoder_initial_state, decoder_initial_context

        for t in range(100):
            word, next_state, next_context = self.words_next_step(
                state=next_state,
                context=next_context,
                raw_query=raw_query,
                query_states=query_encoder_all_states,
                encoder_states=words_encoder_all_states
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

    def encode_rule_tree(self, rule_tree):
        rules, nodes, parent_rules = rule_tree.make_sequence_with_placeholders(self.rules_word_placeholder)
        words_encoder = self.words_model.words_encoder
        feed = words_encoder.placeholders.feed(
            rules=[[rule] for rule in rules],
            nodes=[[node] for node in nodes],
            parent_rules=[[pr] for pr in parent_rules],
            rules_len=[len(rules)]
        )
        fetch = words_encoder.outputs.fetch_all()
        last_state, all_states = self.session.run(fetch, feed)
        return last_state, all_states

    def generate_words_for_rule_tree(self, rule_tree, query, raw_query):
        query_encoder_pc = self.words_model.query_encoder.placeholders
        query_encoder_feed = {
            query_encoder_pc.query_ids: [[token] for token in query],
            query_encoder_pc.query_length: [len(query)],
        }
        query_encoder_all_states = self.session.run(
            fetches=self.words_model.query_encoder.outputs.all_states,
            feed_dict=query_encoder_feed
        )

        last_state, all_states = self.encode_rule_tree(rule_tree)

        words = self.generate_words(
            words_encoder_last_state=last_state,
            words_encoder_all_states=all_states,
            query_encoder_all_states=query_encoder_all_states,
            raw_query=raw_query
        )

        words_for_placement = self.split_words_by_eos(words)

        return words_for_placement

    def generate_rules_for_query(self, query, raw_query):
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

        rule_tree = self.beam_search_rules(
            encoder_last_state=query_encoder_last_state,
            encoder_all_states=query_encoder_all_states,
        )
        return rule_tree

    def generate_code_for_query(self, query, raw_query):
        rule_tree = self.generate_rules_for_query(query, raw_query)
        words = self.generate_words_for_rule_tree(rule_tree, query, raw_query)
        result = self.fill_placeholders(rule_tree, words)
        return result
