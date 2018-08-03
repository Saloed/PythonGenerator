from __future__ import print_function

import numpy as np

from current_net_conf import *
from evaluation.structures import RulesTree, NODE_VALUE_PLACEHOLDER


def rule_trees_next_step(rule_trees, rules_decoder, session):
    decoder_input = rules_decoder.placeholders['rules_decoder_inputs']
    decoder_state = rules_decoder.placeholders['rules_decoder_states']

    fetches = [
        rules_decoder.outputs.rules,
        rules_decoder.outputs.rules_decoder_new_state
    ]

    decoder_inputs = [rt.get_decoder_input() for rt in rule_trees]
    decoder_input_value = np.concatenate(decoder_inputs)

    num_layers = len(decoder_state)

    layer_states = [[] for _ in range(num_layers)]
    for rt in rule_trees:
        states = rt.get_state()
        for i in range(num_layers):
            layer_states[i].append(states[i])

    decoder_state_values = [np.concatenate(st) for st in layer_states]

    feed = {ds: dsv for ds, dsv in zip(decoder_state, decoder_state_values)}
    feed[decoder_input] = decoder_input_value
    rules, new_state = session.run(fetches=fetches, feed_dict=feed)
    new_layer_states = [[[st] for st in states] for states in zip(*new_state)]
    return zip(rule_trees, rules, new_layer_states)


def beam_search_rules(session, rules_decoder, rules_grammar, encoder_last_state, rules_end_marker, rules_count):
    decoder_initial_input = rules_decoder.initializer['rules_decoder_inputs']()
    decoder_initial_state = rules_decoder.initializer['rules_decoder_state'](encoder_last_state)

    rules_tree_root = RulesTree.create_new(rules_grammar)
    rules_tree_root.set_state(decoder_initial_state)
    rules_tree_root.set_decoder_input(decoder_initial_input)
    rules_tree_root.initialize_decoder_input_source(rules_count)

    active_rule_trees = [rules_tree_root]
    completed_rule_trees = []

    for t in range(100):
        rule_trees = rule_trees_next_step(active_rule_trees, rules_decoder, session)

        new_rule_trees = []
        for rule_tree, next_rule_probs, next_state in rule_trees:
            frontier_nt = rule_tree.frontier_nt()
            possible_rules = rules_grammar[frontier_nt.as_type_node]
            for rule in possible_rules:
                rule_id = rules_grammar.rule_to_id[rule]
                score = np.log(next_rule_probs[rule_id])
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

        not_completed_rule_trees.sort(key=lambda rt: -rt.score)
        active_rule_trees = not_completed_rule_trees[:RULES_BEAM_SIZE]

    completed_rule_trees.sort(key=lambda rt: -rt.score)
    result_rule_tree = completed_rule_trees[0]
    return result_rule_tree


def words_next_step(session, words_decoder, words_r_index, query_r_index, query, query_states, state, inputs):
    decoder_input = words_decoder.placeholders['words_decoder_inputs']
    decoder_state = words_decoder.placeholders['words_decoder_state']
    query_encoder_states = words_decoder.placeholders['query_encoder_states']

    decoder_fetches = [
        words_decoder.outputs.words_decoder_new_state,
        words_decoder.outputs.words_logits,
        words_decoder.outputs.copy_scores,
        words_decoder.outputs.generated_tokens,
        words_decoder.outputs.generate_or_copy
    ]

    decoder_feed = {dst: dstv for dst, dstv in zip(decoder_state, state)}

    decoder_feed.update({
        decoder_input: inputs,
        query_encoder_states: query_states
    })

    next_state, next_input, copy_score, gen_token, gen_or_copy = session.run(
        fetches=decoder_fetches,
        feed_dict=decoder_feed
    )

    copy_prob, gen_prob = gen_or_copy[0][0]

    if copy_prob > gen_prob:
        copy_id = np.argmax(copy_score[0][0])
        query_id = query[copy_id]
        result = query_r_index[query_id]
    else:
        gen_id = np.argmax(gen_token[0][0])
        result = words_r_index[gen_id]

    return result, next_state, next_input


def generate_words(session, words_decoder, words_encoder_last_state, query_encoder_all_states, words_end_marker,
                   token_count, words_r_index, query_r_index, query):
    end_marker = words_r_index[words_end_marker]
    decoder_initial_input = words_decoder.initializer['words_decoder_inputs']()
    decoder_initial_state = words_decoder.initializer['words_decoder_state'](words_encoder_last_state)

    result_words = []

    next_state, next_input = decoder_initial_state, decoder_initial_input

    for t in range(100):
        word, next_state, next_input = words_next_step(
            session=session,
            words_decoder=words_decoder,
            words_r_index=words_r_index,
            query_r_index=query_r_index,
            query=query,
            query_states=query_encoder_all_states,
            state=next_state,
            inputs=next_input
        )

        if word == end_marker:
            break

        result_words.append(word)

    return result_words


def join_word_groups(word_groups):
    return [
        ''.join(word_group)
        for word_group in word_groups
    ]


def split_words_by_eos(words):
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

    return join_word_groups(result)


def fill_placeholders(rule_tree, words):
    words_for_placement = split_words_by_eos(words)

    class Context:
        word_id = 0

    def traversal(node):
        if node.value is not None and node.value == NODE_VALUE_PLACEHOLDER:
            if Context.word_id < len(words_for_placement):
                node.value = words_for_placement[Context.word_id]
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


def generate_code_for_query(query, session, stm, rules_grammar, words_r_index, query_r_index, seq_end_markers, counts):
    query_encoder_pc = stm.query_encoder.placeholders

    query_encoder_feed = {
        query_encoder_pc['query_ids']: [[token] for token in query],
        query_encoder_pc['query_length']: [len(query)],
    }

    query_encoder_fetches = [
        stm.query_encoder.outputs.last_state,
        stm.query_encoder.outputs.all_states
    ]

    query_encoder_last_state, query_encoder_all_states = session.run(
        fetches=query_encoder_fetches,
        feed_dict=query_encoder_feed
    )

    rule_tree = beam_search_rules(
        session=session,
        rules_decoder=stm.rules_decoder,
        rules_grammar=rules_grammar,
        encoder_last_state=query_encoder_last_state,
        rules_end_marker=seq_end_markers['rules'],
        rules_count=counts['rules']
    )

    rule_sequence = rule_tree.rules
    rule_id_sequence = [rules_grammar.rule_to_id[r] for r in rule_sequence] + [seq_end_markers['rules']]

    words_encoder_pc = stm.words_encoder.placeholders
    words_encoder_feed = {
        words_encoder_pc['rules_target']: [[rule_id] for rule_id in rule_id_sequence],
        words_encoder_pc['rules_sequence_length']: [len(rule_id_sequence)]
    }
    words_encoder_fetches = [
        stm.words_encoder.outputs.last_state,
        stm.words_encoder.outputs.all_states
    ]

    words_encoder_last_state, words_encoder_all_states = session.run(
        fetches=words_encoder_fetches,
        feed_dict=words_encoder_feed
    )

    words = generate_words(
        session=session,
        words_decoder=stm.words_decoder,
        words_encoder_last_state=words_encoder_last_state,
        query_encoder_all_states=query_encoder_all_states,
        words_end_marker=seq_end_markers['words'],
        token_count=counts['words'],
        words_r_index=words_r_index,
        query_r_index=query_r_index,
        query=query
    )

    result = fill_placeholders(rule_tree, words)
    return result
