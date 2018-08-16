from NL2code.astnode import DecodeTree

NODE_VALUE_PLACEHOLDER = '__<PLACEHOLDER>__'


class RulesTreeRepr:
    @staticmethod
    def get_nodes(node):
        if node is None:
            return []
        return RulesTreeRepr.get_nodes(node.parent) + [node]

    @staticmethod
    def create(rule_tree):
        nodes = RulesTreeRepr.get_nodes(rule_tree)
        reprs = [RulesTreeRepr(node) for node in nodes]
        for parent, node in zip(reprs, reprs[1:]):
            node.parent = parent
            parent.child = node
        return reprs[0]

    def __init__(self, node):
        self.tree = node.tree
        self.score = node.score
        self.rule = node.rule
        self.best_rules = node.best_rules

    def __repr__(self):
        return repr(self.rule)


class RulesTree:
    def __init__(self, grammar, tree, time, has_grammar_error, pc_count):
        self.grammar = grammar
        self.tree = tree
        self.time = time
        self.has_grammar_error = has_grammar_error
        self.placeholders_count = pc_count

        self.score = 0.0

        self.rule_id = None
        self.node_id = None

        self.parent = None

        self.__state = None

        self.__frontier_nt = self.tree
        self.__frontier_nt_t = -1

        self.__view = None
        self.best_rules = None
        self.rule = None

    @property
    def view(self):
        if self.__view is None:
            self.__view = RulesTreeRepr.create(self)
        return self.__view

    @staticmethod
    def create_new(grammar):
        new_rule_tree = RulesTree(grammar, DecodeTree(grammar.root_node.type), -1, False, 0)
        new_rule_tree.node_id = grammar.get_node_type_id(new_rule_tree.tree.type)
        new_rule_tree.rule_id = -1
        return new_rule_tree

    def copy(self):
        return RulesTree(self.grammar, self.tree.copy(), self.time, self.has_grammar_error, self.placeholders_count)

    def apply(self, rule, score):
        new_rule_tree = self.copy()
        new_rule_tree.parent = self
        new_rule_tree.apply_rule(rule)
        new_rule_tree.score = self.score + score
        new_rule_tree.rule_id = self.grammar.rule_to_id[rule]

        if new_rule_tree.is_finished():
            return new_rule_tree

        frontier_nt = new_rule_tree.frontier_nt()
        new_rule_tree.node_id = self.grammar.get_node_type_id(frontier_nt.type)

        if not new_rule_tree.frontier_node_has_value():
            return new_rule_tree

        while new_rule_tree.frontier_node_has_value():

            new_rule_tree.set_value_placeholder()

            if new_rule_tree.is_finished():
                return new_rule_tree

            frontier_nt = new_rule_tree.frontier_nt()
            new_rule_tree.node_id = self.grammar.get_node_type_id(frontier_nt.type)

        return new_rule_tree

    def is_finished(self):
        return self.frontier_nt() is None

    def get_state(self):
        if self.__state is None:
            raise Exception('Decoder state unknown')
        return self.__state

    def set_state(self, state):
        self.__state = state

    def get_parent_id_and_state(self):
        if self.parent is not None:
            return self.parent.rule_id, self.parent.get_state()
        # this node is root
        return self.root_parent_data

    def __repr__(self):
        return self.tree.__repr__()

    def can_expand(self, node):
        if self.grammar.is_value_node(node):
            # if the node is finished
            if node.value is None:
                return True
            if node.value == NODE_VALUE_PLACEHOLDER:
                return False
            raise Exception('Unexpected: node value is undefined')
        elif self.grammar.is_terminal(node):
            return False
        return True

    def apply_rule(self, rule, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        # assert rule.parent.type == nt.type
        if rule.parent.type != nt.type:
            self.has_grammar_error = True

        self.time += 1
        # set the time step when the rule leading by this nt is applied
        nt.t = self.time
        # record the ApplyRule action that is used to expand the current node
        nt.applied_rule = rule

        for child_node in rule.children:
            child = DecodeTree(child_node.type, child_node.label, child_node.value)
            nt.add_child(child)

    def frontier_node_has_value(self):
        frontier_node_type = self.frontier_nt()
        return self.grammar.is_value_node(frontier_node_type)

    def set_value_placeholder(self, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        self.time += 1

        if nt.value is None:
            # this terminal node is empty
            nt.t = self.time
            nt.value = NODE_VALUE_PLACEHOLDER
            self.placeholders_count += 1
        else:
            raise Exception('Unexpected')

    def frontier_nt_helper(self, node):
        if node.is_leaf:
            if self.can_expand(node):
                return node
            else:
                return None

        for child in node.children:
            result = self.frontier_nt_helper(child)
            if result:
                return result

        return None

    def frontier_nt(self):
        if self.__frontier_nt_t == self.time:
            return self.__frontier_nt
        else:
            _frontier_nt = self.frontier_nt_helper(self.tree)
            self.__frontier_nt = _frontier_nt
            self.__frontier_nt_t = self.time

            return _frontier_nt

    def make_sequence_with_placeholders(self, rules_word_pc):
        #  fixme: wtf?????
        # if not self.is_finished():
        #     raise Exception('Dont make sequence for unfinished trees')

        class Context:
            sequence = []

        grammar = self.grammar

        def visit_tree(node):
            rule = node.applied_rule
            rule_id = grammar.rule_to_id[rule] if rule is not None else rules_word_pc
            node_id = get_node_type_id(grammar, node.as_type_node)
            parent = node.parent
            parent_rule_id = grammar.rule_to_id[parent.applied_rule] if parent is not None else -1
            Context.sequence.append((rule_id, node_id, parent_rule_id))

            for child in node.children:
                visit_tree(child)

        visit_tree(self.tree)

        rules = [rule for rule, _, _ in Context.sequence]
        nodes = [node for _, node, _ in Context.sequence]
        parent_rules = [pr for _, _, pr in Context.sequence]
        return rules, nodes, parent_rules


def get_node_type_id(grammar, node):
    #  fixme: tricky hack
    from evaluation.NL2code.astnode import ASTNode
    from NL2code.lang.util import typename

    if isinstance(node, ASTNode):
        type_repr = typename(node.type)
        return grammar.node_type_to_id[type_repr]
    else:
        # assert isinstance(node, str)
        # it is a type
        type_repr = typename(node)
        return grammar.node_type_to_id[type_repr]
