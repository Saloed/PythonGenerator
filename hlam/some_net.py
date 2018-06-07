from some_net_stuff.Structures import Token, Nodes


def _indexate(nodes) -> list:
    for i, node in enumerate(nodes):
        node.index = i
    return nodes


def _tree_to_list(tree_root_token: Token) -> list:
    def convert(token: Token, tokens: list):
        tokens.append(token)
        for child in token.children:
            convert(child, tokens)

    token_list = []
    convert(tree_root_token, token_list)
    return _indexate(token_list)


def convert_to_node(tree):
    root_token = _convert_to_token(tree)
    tree_as_list = _tree_to_list(root_token)
    non_leafs = [t for t in tree_as_list if not t.is_leaf]
    return Nodes(root_token, tree_as_list, non_leafs)


def _convert_to_token(node, parent=None):
    tk = Token(node['type'], parent, False)
    children = [
        _convert_to_token(child, tk)
        for child in node.get('children', [])
    ]
    if not children:
        tk.is_leaf = True
    tk.children = children
    return tk
