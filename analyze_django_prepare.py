import json
import re

import pandas as pd

from collections import defaultdict


COMMON_WORD_COUNT_THRESHOLD = 0

WORD_PLACEHOLDER_TOKEN = '___WORD_PC___'
SUBTREE_START_TOKEN = '('
SUBTREE_END_TOKEN = ')'
SEQUENCE_END_TOKEN = '___SEQ_END___'


def count_items(_list, sort=False):
    items_with_count = defaultdict(int)
    for item in _list:
        items_with_count[item] += 1
    items_with_count = list(items_with_count.items())
    if sort:
        items_with_count.sort(key=lambda it: it[1], reverse=True)
    return items_with_count


def convert_single_ast(ast_str):
    subtree_match = SUBTREE_START_TOKEN + SUBTREE_END_TOKEN
    ast_str = re.sub(r'([%s])' % subtree_match, r' \1 ', ast_str)
    tokens = re.sub(r'\s+', r' ', ast_str).strip().split()
    for i in range(1, len(tokens)):
        if tokens[i - 1] != SUBTREE_START_TOKEN and tokens[i] not in [SUBTREE_START_TOKEN, SUBTREE_END_TOKEN]:
            tokens[i] = WORD_PLACEHOLDER_TOKEN
    tokens.append(SEQUENCE_END_TOKEN)
    return tokens


def convert_ast(data_set_ast):
    return [convert_single_ast(it) for it in data_set_ast]


def convert_words(data_set_words):
    return [words.split() + [SEQUENCE_END_TOKEN] for words in data_set_words]


def strip_word(word):
    return word.strip(',.\'\"')


def split_by_dot(word):
    word = strip_word(word)
    splitted = word.split('.')
    if len(splitted) == 1:
        yield word
    else:
        yield splitted[0]
        for word in splitted[1:]:
            yield '.'
            yield word


def convert_description(data_set_desc):
    return [
        [
            word
            for dsc in desc.split()
            for word in split_by_dot(dsc)
        ] + [SEQUENCE_END_TOKEN]
        for desc in data_set_desc
    ]


def get_ast_token_indexes(data_set_ast):
    flatten_tokens = [tk for ast in data_set_ast for tk in ast]
    counted_tokens = count_items(flatten_tokens, sort=True)
    index = {tk: i for i, (tk, _) in enumerate(counted_tokens)}
    r_index = {i: tk for tk, i in index.items()}
    return index, r_index


def get_indexed_ast(data_set_ast, index):
    return [
        [
            index[tk]
            for tk in ast
        ]
        for ast in data_set_ast
    ]


def get_description_index(data_set_desc):
    flatten_words = [word for desc in data_set_desc for word in desc]
    characters = {ch for word in flatten_words for ch in word}
    counted_words = count_items(flatten_words, sort=False)
    common_words = [word for word, cnt in counted_words if cnt > COMMON_WORD_COUNT_THRESHOLD]
    index = {word: i for i, word in enumerate(set(common_words) | characters)}
    r_index = {i: word for word, i in index.items()}
    return index, r_index


def get_single_description_length(desc):
    return sum(len(word) if isinstance(word, list) else 1 for word in desc)


def get_single_description_with_common_words(desc, index):
    new_desc = [word if word in index else list(word) for word in desc]
    return new_desc  # , get_single_description_length(new_desc)


def get_description_with_common_words(data_set_desc, index):
    return [get_single_description_with_common_words(desc, index) for desc in data_set_desc]


def get_indexed_flatten_description(data_set_desc_with_weight, index):
    return [
        [
            (index[word], weight)
            for word, weight in weighted_desc
        ]
        for weighted_desc in data_set_desc_with_weight
    ]


def get_description_word_flatten_with_weight(word):
    if not isinstance(word, list):
        yield word, 1.0
    else:
        word_weight = 1 / len(word)
        for ch in word:
            yield ch, word_weight


def get_description_flatten_with_weighted_words(data_set_desc):
    return [
        [
            weighted_word
            for word in desc
            for weighted_word in get_description_word_flatten_with_weight(word)
        ]
        for desc in data_set_desc
    ]


def split_flatten_description_and_weights(data_set_desc_with_weight):
    description = [
        [
            word
            for word, _ in desc
        ]
        for desc in data_set_desc_with_weight
    ]
    weights = [
        [
            weight
            for _, weight in desc
        ]
        for desc in data_set_desc_with_weight
    ]
    return description, weights


def get_words_indexes(data_set_words):
    flatten_words = [word for words in data_set_words for word in words]
    counted_words = count_items(flatten_words, sort=False)
    index = {word: i for i, (word, _) in enumerate(counted_words)}
    r_index = {i: word for word, i in index.items()}
    return index, r_index


def get_indexed_words(data_set_words, index):
    return [
        [
            index[word]
            for word in words
        ]
        for words in data_set_words
    ]


def convert_json_content():
    data_set = pd.read_json('django_data_set_str.json')
    converted_data_set = {
        'src': list(data_set['src']),
        'description': convert_description(data_set['description']),
        'ast': convert_ast(data_set['ast']),
        'words': convert_words(data_set['words'])
    }
    ast_token_index, ast_token_r_index = get_ast_token_indexes(converted_data_set['ast'])
    indexed_ast = get_indexed_ast(converted_data_set['ast'], ast_token_index)

    description_word_index, description_word_r_index = get_description_index(converted_data_set['description'])
    description_with_common_words = get_description_with_common_words(
        data_set_desc=converted_data_set['description'],
        index=description_word_index,
    )
    weighted_flatten_description = get_description_flatten_with_weighted_words(description_with_common_words)
    indexed_weighted_flatten_description = get_indexed_flatten_description(
        data_set_desc_with_weight=weighted_flatten_description,
        index=description_word_index,
    )
    indexed_description, description_word_weights = split_flatten_description_and_weights(
        data_set_desc_with_weight=indexed_weighted_flatten_description
    )

    words_index, words_r_index = get_words_indexes(converted_data_set['words'])
    indexed_words = get_indexed_words(converted_data_set['words'], words_index)

    converted_data_set['ast_token_index'] = ast_token_index
    converted_data_set['ast_token_r_index'] = ast_token_r_index
    converted_data_set['indexed_ast'] = indexed_ast

    converted_data_set['desc_word_index'] = description_word_index
    converted_data_set['desc_word_r_index'] = description_word_r_index
    converted_data_set['indexed_description'] = indexed_description
    converted_data_set['description_weights'] = description_word_weights

    converted_data_set['words_index'] = words_index
    converted_data_set['words_r_index'] = words_r_index
    converted_data_set['indexed_words'] = indexed_words

    with open('django_data_set_2.json', 'w') as f:
        json.dump(converted_data_set, f, indent=4)


if __name__ == '__main__':
    convert_json_content()
