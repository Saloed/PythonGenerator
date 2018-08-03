import numpy as np

from NL2code.astnode import DecodeTree

NODE_VALUE_PLACEHOLDER = '__<PLACEHOLDER>__'


class RulesTree:
    def __init__(self, grammar, tree, time, has_grammar_error, pc_count, rules, dis=None):
        self.grammar = grammar
        self.tree = tree
        self.time = time
        self.has_grammar_error = has_grammar_error
        self.placeholders_count = pc_count

        self.score = 0.0
        self.node_id = None
        self.parent_rule_id = None

        self.rules = rules

        self.__state = None

        self.__frontier_nt = self.tree
        self.__frontier_nt_t = -1

        self.__decoder_inputs_source = dis
        self.__decoder_inputs = None

    @staticmethod
    def create_new(grammar):
        new_rule_tree = RulesTree(grammar, DecodeTree(grammar.root_node.type), -1, False, 0, [])
        new_rule_tree.node_id = grammar.get_node_type_id(new_rule_tree.tree.type)
        new_rule_tree.parent_rule_id = -1
        return new_rule_tree

    def initialize_decoder_input_source(self, rules_count):
        self.__decoder_inputs_source = np.eye(rules_count)

    def copy(self):
        return RulesTree(self.grammar, self.tree.copy(), self.time, self.has_grammar_error, self.placeholders_count,
                         self.rules[:], self.__decoder_inputs_source)

    def apply(self, rule, score):
        new_rule_tree = self.copy()
        new_rule_tree.apply_rule(rule)
        new_rule_tree.score = self.score + score

        if new_rule_tree.is_finished():
            return new_rule_tree

        frontier_nt = new_rule_tree.frontier_nt()
        new_rule_tree.node_id = self.grammar.get_node_type_id(frontier_nt.type)
        new_rule_tree.parent_rule_id = self.grammar.rule_to_id[frontier_nt.parent.applied_rule]

        if not new_rule_tree.frontier_node_has_value():
            return new_rule_tree

        while new_rule_tree.frontier_node_has_value():

            new_rule_tree.set_value_placeholder()

            if new_rule_tree.is_finished():
                return new_rule_tree

            frontier_nt = new_rule_tree.frontier_nt()
            new_rule_tree.node_id = self.grammar.get_node_type_id(frontier_nt.type)
            new_rule_tree.parent_rule_id = self.grammar.rule_to_id[frontier_nt.parent.applied_rule]

        return new_rule_tree

    def is_finished(self):
        return self.frontier_nt() is None

    def get_state(self):
        if self.__state is None:
            raise Exception('Decoder state unknown')
        return self.__state

    def set_state(self, state):
        self.__state = state

    def get_decoder_input(self):
        if self.__decoder_inputs is None:
            self.__decoder_inputs = [self.__decoder_inputs_source[self.node_id]]
        return self.__decoder_inputs

    def set_decoder_input(self, new_input):
        self.__decoder_inputs = new_input

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
        self.rules.append(rule)

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

    def get_action_parent_t(self):
        """
        get the time step when the parent of the current
        action was generated
        WARNING: 0 will be returned if parent if None
        """
        nt = self.frontier_nt()

        if nt.parent:
            return nt.parent.t
        else:
            return 0
