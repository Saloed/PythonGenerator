import json

from analyze_django_prepare import SUBTREE_START_TOKEN
from prepare_ast_to_tree import convert_trees_to_node


def main():
    with open('django_data_set_3.json') as f:
        data_set = json.load(f)
    trees = data_set['ast_tree']
    split_position = len(trees) // 10
    trees = trees[split_position:]
    nodes = convert_trees_to_node(trees, data_set['ast_token_index'][SUBTREE_START_TOKEN])
    import ipdb
    ipdb.set_trace()
    a = 3


if __name__ == '__main__':
    main()
