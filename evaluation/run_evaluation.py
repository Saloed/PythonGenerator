from __future__ import print_function
from __future__ import division

import json
import pickle

import astor
import tensorflow as tf

import batching
from NL2code.lang.py import parse
from evaluation.code_evaluator import CodeEvaluator
from evaluation.generate_code import CodeGenerator
from model.model_single_step import build_single_step_rules_model, build_single_step_words_model
from model.model_single_step import get_rules_variables, get_words_variables

from current_net_conf import *


def decode_tree_to_python_ast(decode_tree):
    from NL2code.lang.py.unaryclosure import compressed_ast_to_normal

    compressed_ast_to_normal(decode_tree)
    decode_tree = decode_tree.children[0]
    terminals = decode_tree.get_leaves()

    for terminal in terminals:
        if terminal.value is not None and type(terminal.value) is str:
            if terminal.value.endswith('<eos>'):
                terminal.value = terminal.value[:-5]

        if terminal.type in {int, float, str, bool}:
            # cast to target data type
            try:
                terminal.value = terminal.type(terminal.value)
            except:
                terminal.value = terminal.type('0')

    ast_tree = parse.parse_tree_to_python_ast(decode_tree)

    return ast_tree


def generate_code_for_sample(sample, test_set_examples, evaluator, code_generator):
    sample_id, query = sample[0], sample[1]

    test_example = test_set_examples[sample_id]
    raw_query = test_example.query

    sample_decode_tree = code_generator.generate_code_for_query(query, raw_query)
    try:
        sample_ast = decode_tree_to_python_ast(sample_decode_tree)
        sample_code = astor.to_source(sample_ast).strip()
    except Exception as ex:
        print(ex)
        return

    acc, bleu = evaluator.eval(sample_code, test_example)
    if not acc:
        print(sample_id)
        print(acc, bleu)
        print(evaluator.normalize_code(sample_code, test_example))
        print(test_example.meta_data['raw_code'])
        print('---------------------------')
    return sample_id, sample_code


def generate_code_for_data_set(data_set, **kwargs):
    return [
        generate_code_for_sample(sample, **kwargs)
        for sample in data_set
    ]


data_set_type = 'valid'  # test


def evaluate():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    with open(DATA_SET_BASE_DIR + FULL_DATA_SET_NAME) as f:
        full_data_set = pickle.load(f)

    test_set = batching.construct_rules_data_set(data_set[data_set_type])

    _train_set, _valid_set, _test_set = full_data_set
    full_test_set = {'train': _train_set, 'valid': _valid_set, 'test': _test_set}[data_set_type]

    test_set_examples = {ex.raw_id: ex for ex in full_test_set.examples}

    num_rule_tokens = data_set['train']['rules_tokens_count']
    num_query_tokens = data_set['train']['query_tokens_count']
    num_word_tokens = data_set['train']['words_size']
    num_rule_nodes = data_set['train']['rules_nodes_count']

    query_end_marker = data_set['train']['query_seq_end']
    rules_end_marker = data_set['train']['rules_seq_end']
    word_end_marker = data_set['train']['words_seq_end']

    rules_model = build_single_step_rules_model(num_query_tokens, num_rule_tokens, num_rule_nodes)
    words_model = build_single_step_words_model(num_query_tokens, num_rule_tokens, num_rule_nodes, num_word_tokens)

    grammar = full_test_set.grammar
    words_index = data_set['train']['words_index']
    words_r_index = {int(value): key for key, value in words_index.items()}

    end_markers = {
        'query': query_end_marker,
        'rules': rules_end_marker,
        'words': word_end_marker
    }

    rules_word_placeholder = data_set[data_set_type]['rules_word_pc']

    counts = {
        'query': num_query_tokens,
        'rules': num_rule_tokens,
        'words': num_word_tokens,
        'nodes': num_rule_nodes
    }

    evaluator = CodeEvaluator()

    rules_model_saver = tf.train.Saver(get_rules_variables())
    words_model_saver = tf.train.Saver(get_words_variables())
    rules_model_name = MODEL_SAVE_PATH + BEST_RULES_MODEL_BASE_NAME
    words_model_name = MODEL_SAVE_PATH + BEST_WORDS_MODEL_BASE_NAME

    with tf.Session() as session:
        rules_model_saver.restore(session, rules_model_name)
        words_model_saver.restore(session, words_model_name)

        code_generator = CodeGenerator(
            session=session,
            rules_model=rules_model,
            words_model=words_model,
            rules_grammar=grammar,
            words_r_index=words_r_index,
            seq_end_markers=end_markers,
            counts=counts,
            rules_word_placeholder=rules_word_placeholder
        )

        code_for_data_set = generate_code_for_data_set(
            test_set,
            test_set_examples=test_set_examples,
            evaluator=evaluator,
            code_generator=code_generator
        )

    with open('tmp/generated_code', 'wb') as f:
        pickle.dump(code_for_data_set, f)

    print('RRREEESSS')
    print(evaluator.get_accuracy(), evaluator.get_bleu())
