import ast

import astor.code_gen as astor_gen
import astor.source_repr as astor_src_repr
import astor.op_util as astor_op
import astor.node_util as astor_util


class SourceGenerator(astor_gen.SourceGenerator):
    def visit_EmptyNode(self, node):
        self.write('__EMPTY_NODE__')

    def visit(self, node, abort=astor_util.ExplicitNodeVisitor.abort_visit):
        try:
            super(SourceGenerator, self).visit(node, abort)
        except Exception:
            self.write('__ERROR_NODE__')

    def visit_Assign(self, node):
        if not isinstance(node.targets, (tuple, list)):
            node.targets = (node.targets,)
        return super(SourceGenerator, self).visit_Assign(node)

    def visit_TryExcept(self, node):
        if not isinstance(node.handlers, (tuple, list)):
            node.handlers = (node.handlers,)
        return super(SourceGenerator, self).visit_TryExcept(node)

    # new for Python 3.3
    def visit_Try(self, node):
        if not isinstance(node.handlers, (tuple, list)):
            node.handlers = (node.handlers,)
        return super(SourceGenerator, self).visit_Try(node)


astor_gen.SourceGenerator = SourceGenerator

old_split_lines = astor_src_repr.split_lines


def split_lines(source, maxline=79):
    try:
        return old_split_lines(source, maxline)
    except Exception:
        source = [str(it) for it in source]
        return old_split_lines(source, maxline)


astor_src_repr.split_lines = split_lines

precedence_data = astor_op.precedence_data


def get_op_precedence(obj, precedence_data=precedence_data, type=type):
    return precedence_data.get(type(obj), 0)


astor_op.get_op_precedence = get_op_precedence
astor_gen.get_op_precedence = get_op_precedence


class EmptyNode(ast.AST):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __len__(self):
        return 0

    def __radd__(self, other):
        return other

    def __getitem__(self, item):
        return EmptyNode()
