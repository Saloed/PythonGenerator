import json
from analyze_django_prepare import SEQUENCE_END_TOKEN


# {
#     'rules_count': len(data_set.grammar),
#
#     'data': {ex.raw_id: ex.data for ex in samples},
#     'trees': [get_example_tree(ex) for ex in samples],
#
#     'annot_index': annot_vocab.token_id_map,
#     'annot_r_index': annot_vocab.id_token_map,
#
#     'terminal_index': terminal_vocab.token_id_map,
#     'terminal_r_index': terminal_vocab.id_token_map,
#
# }

def make_tree_token_index(trees):
    _tokens = set()

    def tokens(node):
        nonlocal _tokens
        node, children = node
        _tokens |= {node}
        for child in children:
            tokens(child)

    for tree in trees:
        tokens(tree)

    return {tk: i for i, tk in enumerate(sorted(_tokens))}


def indexate_trees(trees, index):
    def indexate_tree(root):
        node, children = root
        node = index[node]
        children = [indexate_tree(child) for child in children]
        return node, children

    return [indexate_tree(tree) for tree in trees]


def rework(data_set):
    annot_index = data_set['annot_index']
    annot_end_id = len(annot_index)
    annot_index[SEQUENCE_END_TOKEN] = annot_end_id
    words_index = data_set['terminal_index']
    words_end_id = len(words_index)
    words_index[SEQUENCE_END_TOKEN] = words_end_id
    rules_count = data_set['rules_count']
    rules_end_id = rules_count
    rules_count += 1
    words_count = len(words_index)
    annot_count = len(annot_index)
    descriptions = []
    rules = []
    words = []
    copy_words = []
    copy_targets = []

    for query_ids, rule_ids, word_ids, query_terminal_ids, copy_ids in data_set['data']:
        query_terminal_ids = [qti if qti != 1 else -1 for qti in query_terminal_ids]
        query_terminal_ids = query_terminal_ids[:len(query_ids)]
        query_terminal_ids.append(-1)

        query_ids.append(annot_end_id)
        rule_ids.append(rules_end_id)
        word_ids.append(words_end_id)
        copy_ids.append(-1)

        descriptions.append(query_ids)
        rules.append(rule_ids)
        words.append(word_ids)
        copy_words.append(query_terminal_ids)
        copy_targets.append(copy_ids)

    trees = data_set['trees']
    tree_index = make_tree_token_index(trees)
    trees = indexate_trees(trees, tree_index)
    tree_size = len(tree_index)

    return {
        'descriptions': descriptions,
        'desc_index': annot_index,
        'desc_seq_end': annot_end_id,
        'desc_size': annot_count,

        'rules': rules,
        'rules_seq_end': rules_end_id,
        'rules_size': rules_count,

        'words': words,
        'words_index': words_index,
        'words_seq_end': words_end_id,
        'words_size': words_count,

        'copy_words': copy_words,
        'copy_targets': copy_targets,

        'trees': trees,
        'tree_size': tree_size,

        'ids': data_set['ids']
    }


HS = True


def main():
    if HS:
        file_name = 'hs_data_set_rules'
        res_file_name = 'hs_data_set_x'
    else:
        file_name = 'django_data_set_rules'
        res_file_name = 'django_data_set_x'

    with open('../django_data_set/data/' + file_name) as f:
        result = json.load(f)
    reworked = {}
    for set_name, data_set in result.items():
        reworked[set_name] = rework(data_set)

    with open(res_file_name, 'w') as f:
        json.dump(reworked, f)


if __name__ == '__main__':
    main()
