from collections import namedtuple

Nodes = namedtuple('Nodes', ['root_node', 'all_nodes', 'non_leafs'])


class Token:
    def __init__(self, token_type, parent, is_leaf, pos=0, start_line=None, end_line=None, author=None):
        self.token_type = token_type
        self.parent = parent
        self.children = []
        self.is_leaf = is_leaf
        self.pos = pos
        self.left_rate = 0
        self.right_rate = 0
        self.leaf_num = 0
        self.children_num = 0
        self.index = None
        self.start_line = start_line
        self.end_line = end_line
        self.author = author

    def __str__(self):
        return str(self.token_type) + '_' + str(self.index)

    def __repr__(self):
        return self.__str__()