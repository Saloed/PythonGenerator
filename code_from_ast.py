import re
import ast
import astor
import json
import builtins

from utilss import fix_r_index_keys, sequence_to_tree
from analyze_django_prepare import SEQUENCE_END_TOKEN, WORD_PLACEHOLDER_TOKEN, SUBTREE_START_TOKEN, SUBTREE_END_TOKEN

import code_generator

BOOL_TYPES = {'bool'}
NUMERIC_TYPES = {'int', 'float', 'complex'}
STR_TYPES = {'str'}
BINARY_TYPES = {'bytes', 'bytearray', 'memoryview'}
BUILT_IN_TYPES = BOOL_TYPES | NUMERIC_TYPES | STR_TYPES | BINARY_TYPES

constructor_exception_re = 'constructor takes either (\w+) or (\w+)'


def get_ast_node(node_type, default_empty, *args, **kwargs):
    ast_node_type = getattr(ast, node_type)
    _args = []
    args_iter = iter(args)

    def get_default():
        return [] if not default_empty else code_generator.EmptyNode()

    for field_name in ast_node_type._fields:
        if field_name in kwargs:
            _args.append(kwargs[field_name])
        else:
            elem = next(args_iter, get_default())
            _args.append(elem)
    # try:
    ast_node = ast_node_type(*_args)
    # except Exception as ex:
    #     groups = re.search(constructor_exception_re, str(ex))
    #     min_args = int(groups.group(1))
    #     max_args = int(groups.group(2))
    #     current_len = len(args) + len(kwargs)
    #     for i in range(max_args - current_len):
    #         args = args + (code_generator.EmptyNode(),)
    #     ast_node = ast_node_type(*args, **kwargs)

    for field_name in ast_node._fields:
        attr = getattr(ast_node, field_name)
        if isinstance(attr, code_generator.EmptyNode):
            if field_name == 'ctx':
                default_value = ast.Load()
            else:
                continue
            setattr(ast_node, field_name, default_value)
    return ast_node


def convert_to_ast(code, words, default_empty):
    word_iter = iter(words)

    def walk_and_insert_word_pc(node):
        nonlocal word_iter
        node_type, children = node
        if node_type == WORD_PLACEHOLDER_TOKEN:
            return next(word_iter, '__NO_WORD__')
        children = [walk_and_insert_word_pc(child) for child in children]
        children = [child for child in children if child != '__EMPTY__CHILDREN__']
        if node_type == 'list':
            return children
        if hasattr(ast, node_type) and not (
                (node_type in ['arg', 'keyword']) and not (len(children) == 0 or len(children) == 2)
        ) and node_type != 'slice':
            kwargs = {child[0]: child[1] for child in children if isinstance(child, tuple)}
            args = [child for child in children if not isinstance(child, tuple)]
            ast_node = get_ast_node(node_type, default_empty, *args, **kwargs)
            return ast_node
        if not children:
            return '__EMPTY__CHILDREN__'
        if len(children) == 1:
            children = children[0]
        if node_type == 'NoneType':
            return None
        if node_type in BUILT_IN_TYPES:
            type_class = getattr(builtins, node_type)
            if node_type == 'bytes':
                return type_class(children, 'utf-8')
            return type_class(children)
        return node_type, children

    result = [walk_and_insert_word_pc(root) for root in code]
    return result


def generate_source(asts, words, ast_with_bugs=False):
    trees = [
        sequence_to_tree(sample, SEQUENCE_END_TOKEN, SUBTREE_START_TOKEN, SUBTREE_END_TOKEN, WORD_PLACEHOLDER_TOKEN)
        for sample in asts
    ]

    converted = [
        convert_to_ast(code, word, ast_with_bugs)
        for code, word in zip(trees, words)
    ]
    sources = [[astor.to_source(root) for root in tree] for tree in converted]
    sources = ['\n'.join(src_parts) for src_parts in sources]
    return sources


def main():
    with open('django_data_set_4.json') as f:
        data_set = json.load(f)
    ast_r_index = fix_r_index_keys(data_set['ast_token_r_index'])
    words_r_index = fix_r_index_keys(data_set['words_r_index'])
    train_asts = data_set['train']['indexed_ast'] + data_set['valid']['indexed_ast'] + data_set['test']['indexed_ast']
    train_words = data_set['train']['indexed_words'] + data_set['valid']['indexed_words'] + data_set['test'][
        'indexed_words']
    train_asts = [[ast_r_index[it] for it in indexed_ast] for indexed_ast in train_asts]
    train_words = [[words_r_index[it] for it in indexed_words] for indexed_words in train_words]
    sources = generate_source(train_asts, train_words)
    ok_len = len([src for src in sources if src])
    print(ok_len == len(train_asts))


if __name__ == '__main__':
    main()
