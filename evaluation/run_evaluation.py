import json
import pickle

import astor
import tensorflow as tf

import batching
from NL2code.lang.py import parse
from evaluation.code_evaluator import CodeEvaluator
from evaluation.generate_code import generate_code_for_query
from model.model_single_step import build_single_step_model

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


def generate_code_for_sample(sample, test_set_examples, evaluator, **kwargs):
    sample_id, query = sample[0], sample[1]
    sample_decode_tree = generate_code_for_query(query, **kwargs)
    sample_ast = decode_tree_to_python_ast(sample_decode_tree)
    sample_code = astor.to_source(sample_ast).strip()

    test_example = test_set_examples[sample_id]
    acc, bleu = evaluator.eval(sample_code, test_example)
    print(acc, bleu)
    print(sample_code)
    print(test_example.code)
    return sample_id, sample_code


def generate_code_for_data_set(data_set, **kwargs):
    return [
        generate_code_for_sample(sample, **kwargs)
        for sample in data_set
    ]


def evaluate():
    with open(DATA_SET_BASE_DIR + DATA_SET_NAME) as f:
        data_set = json.load(f)

    with open(DATA_SET_BASE_DIR + FULL_DATA_SET_NAME) as f:
        full_data_set = pickle.load(f)

    test_set = batching.construct_data_set(**{
        field_name: data_set['test'][DATA_SET_FIELD_NAMES_MAPPING[field_name]]
        for field_name in batching.DATA_SET_FIELDS
    })
    _, _, full_test_set = full_data_set

    test_set_examples = {ex.raw_id: ex for ex in full_test_set.examples}

    num_rule_tokens = data_set['train']['rules_tokens_count']
    num_query_tokens = data_set['train']['query_tokens_count']
    num_word_tokens = data_set['train']['words_size']

    query_end_marker = data_set['train']['query_seq_end']
    rules_end_marker = data_set['train']['rules_seq_end']
    word_end_marker = data_set['train']['words_seq_end']

    single_step_model = build_single_step_model(num_query_tokens, num_rule_tokens, num_word_tokens)

    grammar = full_test_set.grammar
    words_index = data_set['test']['words_index']
    query_index = data_set['test']['query_index']
    words_r_index = {int(value): key for key, value in words_index.items()}
    query_r_index = {int(value): key for key, value in query_index.items()}

    end_markers = {
        'query': query_end_marker,
        'rules': rules_end_marker,
        'words': word_end_marker
    }

    counts = {
        'query': num_query_tokens,
        'rules': num_rule_tokens,
        'words': num_word_tokens
    }

    evaluator = CodeEvaluator()

    model_saver = tf.train.Saver()
    model_name = MODEL_SAVE_PATH + MODEL_BASE_NAME

    with tf.Session() as session:
        model_saver.restore(session, model_name)

        code_for_data_set = generate_code_for_data_set(
            test_set,
            test_set_examples=test_set_examples,
            evaluator=evaluator,
            session=session,
            stm=single_step_model,
            rules_grammar=grammar,
            words_r_index=words_r_index,
            query_r_index=query_r_index,
            seq_end_markers=end_markers,
            counts=counts
        )

    with open('tmp/generated_code', 'wb') as f:
        pickle.dump(code_for_data_set, f)

    print('RRREEESSS')
    print(evaluator.get_accuracy(), evaluator.get_bleu())
