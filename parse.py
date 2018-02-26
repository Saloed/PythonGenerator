import ast
import html
import re
from collections import defaultdict

import timeout_decorator as td
import pandas as pd
from multiprocessing import Pool

import sys

DICT_EXPAND_KEY = 'DICT_EXPAND_KEY'


@td.timeout(10)
def parse_body(body):
    codez = re.finditer(r"<p>(.*)</p>\s+<pre[^>]*>[^<]*<code[^>]*>((?:\s|[^<]|<span[^>]*>[^<]+</span>)*)</code></pre>",
                        body)
    codez = map(lambda x: (x.group(1), x.group(2)), codez)
    for message, code in sorted(codez, key=lambda x: len(x), reverse=True):
        # fetch that code
        code = html.unescape(code)
        code = re.sub(r"<[^>]+>([^<]*)<[^>]*>", "\1", code)
        try:
            ast.parse(code)
            return message, code
        except:
            pass
    return None, None


def parse_question_with_answer(qanda):
    q, a = qanda
    try:
        q = parse_body(q)
        a = parse_body(a)
    except Exception:
        return (None, None), (None, None)
    return q, a


def parse(name):
    data = pd.read_csv(name)
    data_to_parse = list(sorted(zip(data['question'], data['answer']), key=lambda x: len(x[0]) + len(x[1])))
    num_tasks = len(data_to_parse)
    parsed = []
    with Pool(8) as pool:
        for i, result in enumerate(pool.imap_unordered(parse_question_with_answer, data_to_parse), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_tasks))
            parsed.append(result)
    parsed = [(q_text, q_code, a_text, a_code) for ((q_text, q_code), (a_text, a_code)) in parsed]
    parsed = pd.DataFrame(data=parsed, columns=['question_text', 'question_code', 'answer_text', 'answer_code'])
    parsed.to_csv('ParsedData.csv')


def set_parents_in_tree(root):
    def set_parent(node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
        for child in ast.iter_child_nodes(node):
            set_parent(child)

    set_parent(root)


class NodePereebator(ast.NodeVisitor):
    def __init__(self, variable_names, function_names):
        super(NodePereebator, self).__init__()
        self.attribute_names = variable_names
        self.function_names = function_names

    def visit_Name(self, node: ast.Name):
        if isinstance(node.parent, ast.Call) and getattr(node.parent, 'func', None) == node:
            self.function_names[node.parent] = node.id

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.parent, ast.Call) and getattr(node.parent, 'func', None) == node:
            self.attribute_names[node.parent] = node.attr


class NodePerehuyator(ast.NodeTransformer):
    def __init__(self, attribute_names, function_names):
        super(NodePerehuyator, self).__init__()
        self.common_functions = function_names
        self.common_attributes = attribute_names
        self.function_replace = {}
        self.attribute_replace = {}

    def visit_Name(self, node: ast.Name):
        if node.id not in self.common_functions:
            if node.id not in self.function_replace:
                self.function_replace[node.id] = len(self.function_replace)
            node.id = self.function_replace[node.id]

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr not in self.common_attributes:
            if node.attr not in self.attribute_replace:
                self.attribute_replace[node.attr] = len(self.attribute_replace)
            node.attr = self.attribute_replace[node.attr]


class NodeUebator(ast.NodeTransformer):
    def visit_AnnAssign(self, node: ast.AnnAssign):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': 'Assign',
            'targets': [node.target],
            'value': node.value,
        }

    def visit_Assert(self, node: ast.Assert):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'msg': node.msg,
        }

    def visit_Assign(self, node: ast.Assign):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'targets': node.targets,
            'value': node.value,
        }

    def visit_AsyncFor(self, node: ast.AsyncFor):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_AsyncWith(self, node: ast.AsyncWith):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Attribute(self, node: ast.Attribute):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'attr': node.attr,
        }

    def visit_AugAssign(self, node: ast.AugAssign):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'target': node.target,
            'value': node.value,
            'op': node.op,
        }

    def visit_AugLoad(self, node: ast.AugLoad):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_AugStore(self, node: ast.AugStore):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Await(self, node: ast.Await):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_BinOp(self, node: ast.BinOp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.op['type'],
            'left': node.left,
            'right': node.right,
        }

    def visit_BoolOp(self, node: ast.BoolOp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.op['type'],
            'values': list(node.values),
        }

    def visit_Break(self, node: ast.Break):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Bytes(self, node: ast.Bytes):
        return {
            'type': node.__class__.__name__,
            'value': str(node.s),
        }

    def visit_Call(self, node: ast.Call):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'func': node.func,
            'args': node.args,
            'keywords': node.keywords,
        }

    def visit_ClassDef(self, node: ast.ClassDef):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Compare(self, node: ast.Compare):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'left': node.left,
            'ops': node.ops,
            'comparators': node.comparators,
        }

    def visit_Constant(self, node: ast.Constant):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Continue(self, node: ast.Continue):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Del(self, node: ast.Del):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Delete(self, node: ast.Delete):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'targets': node.targets,
        }

    def visit_Dict(self, node: ast.Dict):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'values': node.values,
            'keys': [key or DICT_EXPAND_KEY for key in node.keys]
        }

    def visit_DictComp(self, node: ast.DictComp):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Ellipsis(self, node: ast.Ellipsis):
        return {
            'type': node.__class__.__name__,
        }

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'exception_type': node.type,
            'body': node.body,
            'name': node.name,
        }

    def visit_Expr(self, node: ast.Expr):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Expression(self, node: ast.Expression):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_ExtSlice(self, node: ast.ExtSlice):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'dims': node.dims,
        }

    def visit_For(self, node: ast.For):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'target': node.test,
            'iter': node.iter,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_FormattedValue(self, node: ast.FormattedValue):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'format': node.format_spec or {
                'type': 'Str',
                'value': '',
            }
        }

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'name': {
                'type': 'Name',
                'value': node.name,
            },
            'args': node.args,
            'body': node.body,
            'decorators': node.decorator_list,
        }

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'element': node.elt,
            'generators': node.generators,
        }

    def visit_Global(self, node: ast.Global):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_If(self, node: ast.If):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_IfExp(self, node: ast.IfExp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_Import(self, node: ast.Import):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'names': node.names,
        }

    def visit_ImportFrom(self, node: ast.ImportFrom):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'names': node.names,
            'module': node.module,
            'level': str(node.level),
        }

    def visit_Index(self, node: ast.Index):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Interactive(self, node: ast.Interactive):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_JoinedStr(self, node: ast.JoinedStr):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'values': node.values,
        }

    def visit_Lambda(self, node: ast.Lambda):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'args': node.args,
            'body': node.body,
        }

    def visit_List(self, node: ast.List):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_ListComp(self, node: ast.ListComp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'element': node.elt,
            'generators': node.generators,
        }

    def visit_Load(self, node: ast.Load):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Module(self, node: ast.Module):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Name(self, node: ast.Name):
        return {
            'type': node.__class__.__name__,
            'value': node.id,
        }

    def visit_NameConstant(self, node: ast.NameConstant):
        return {
            'type': node.__class__.__name__,
            'value': str(node.value),
        }

    def visit_Nonlocal(self, node: ast.Nonlocal):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Num(self, node: ast.Num):
        return {
            'type': node.__class__.__name__,
            'value': str(node.n),
        }

    def visit_Param(self, node: ast.Param):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Pass(self, node: ast.Pass):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Raise(self, node: ast.Raise):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'exc': node.exc,
            'cause': node.cause,
        }

    def visit_Return(self, node: ast.Return):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Set(self, node: ast.Set):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_SetComp(self, node: ast.SetComp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'element': node.elt,
            'generators': node.generators,
        }

    def visit_Slice(self, node: ast.Slice):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'lower': node.lower,
            'upper': node.upper,
            'step': node.step,
        }

    def visit_Starred(self, node: ast.Starred):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Store(self, node: ast.Store):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Str(self, node: ast.Str):
        return {
            'type': node.__class__.__name__,
            'value': str(node.s),
        }

    def visit_Subscript(self, node: ast.Subscript):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'slice': node.slice,
            'operation': node.ctx,
        }

    def visit_Suite(self, node: ast.Suite):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_Try(self, node: ast.Try):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'body': node.body,
            'handlers': node.handlers,
            'else': node.orelse,
            'final': node.finalbody,
        }

    def visit_Tuple(self, node: ast.Tuple):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.op['type'],
            'operand': node.operand,
        }

    def visit_While(self, node: ast.While):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_With(self, node: ast.With):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'items': node.items,
            'body': node.body,
        }

    def visit_Yield(self, node: ast.Yield):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_YieldFrom(self, node: ast.YieldFrom):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_alias(self, node: ast.alias):
        return {
            'type': node.__class__.__name__,
            'name': node.name,
            'asname': node.asname,
        }

    def visit_arg(self, node: ast.arg):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_arguments(self, node: ast.arguments):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            # todo: aaaaaaaaaaaaaaaaaa
        }

    def visit_comprehension(self, node: ast.comprehension):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'iter': node.iter,
            'target': node.target,
            'ifs': node.ifs,
        }

    def visit_excepthandler(self, node: ast.excepthandler):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_expr(self, node: ast.expr):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_expr_context(self, node: ast.expr_context):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_keyword(self, node: ast.keyword):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'arg': {
                'type': 'Name',
                'value': node.arg,
            },
            'value': node.value,
        }

    def visit_mod(self, node: ast.mod):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_slice(self, node: ast.slice):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_stmt(self, node: ast.stmt):
        node = super(NodeUebator, self).generic_visit(node)
        pass

    def visit_withitem(self, node: ast.withitem):
        node = super(NodeUebator, self).generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'context_expr': node.context_expr,
            'optional_vars': node.optional_vars,
        }

    def visit_Add(self, node):
        return self.visit_operator(node)

    def visit_BitAnd(self, node):
        return self.visit_operator(node)

    def visit_BitOr(self, node):
        return self.visit_operator(node)

    def visit_BitXor(self, node):
        return self.visit_operator(node)

    def visit_Div(self, node):
        return self.visit_operator(node)

    def visit_FloorDiv(self, node):
        return self.visit_operator(node)

    def visit_LShift(self, node):
        return self.visit_operator(node)

    def visit_MatMult(self, node):
        return self.visit_operator(node)

    def visit_Mod(self, node):
        return self.visit_operator(node)

    def visit_Mult(self, node):
        return self.visit_operator(node)

    def visit_Pow(self, node):
        return self.visit_operator(node)

    def visit_RShift(self, node):
        return self.visit_operator(node)

    def visit_Sub(self, node):
        return self.visit_operator(node)

    def visit_operator(self, node: ast.operator):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Invert(self, node):
        return self.visit_unaryop(node)

    def visit_Not(self, node):
        return self.visit_unaryop(node)

    def visit_UAdd(self, node):
        return self.visit_unaryop(node)

    def visit_USub(self, node):
        return self.visit_unaryop(node)

    def visit_unaryop(self, node: ast.unaryop):
        return {
            'type': node.__class__.__name__,
        }

    def visit_And(self, node):
        return self.visit_boolop(node)

    def visit_Or(self, node):
        return self.visit_boolop(node)

    def visit_boolop(self, node):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Eq(self, node):
        return self.visit_cmpop(node)

    def visit_Gt(self, node):
        return self.visit_cmpop(node)

    def visit_GtE(self, node):
        return self.visit_cmpop(node)

    def visit_In(self, node):
        return self.visit_cmpop(node)

    def visit_Is(self, node):
        return self.visit_cmpop(node)

    def visit_IsNot(self, node):
        return self.visit_cmpop(node)

    def visit_Lt(self, node):
        return self.visit_cmpop(node)

    def visit_LtE(self, node):
        return self.visit_cmpop(node)

    def visit_NotEq(self, node):
        return self.visit_cmpop(node)

    def visit_NotIn(self, node):
        return self.visit_cmpop(node)

    def visit_cmpop(self, node):
        return {
            'type': node.__class__.__name__,
        }


def transform_tree(tree):
    pass


def extract_common_names(root, text):
    # ids = {}
    # names = {}
    # def extract_fields(node):
    #     result = {'node': node.__class__.__name__, 'fields': {}}
    #     for field_name, value in ast.iter_fields(node):
    #         if field_name in {'id', 'name', 'arg', 'ctx'}:
    #             if field_name == 'id':
    #                 if value not in ids:
    #                     ids[value] = max(ids.values()) + 1
    #                 value = ids[value]
    #             elif field_name == 'ctx':
    #                 value = value.__class__.__name__
    #             result['fields'][field_name] = value
    #     return result
    try:
        attribute_names, function_names = {}, {}
        set_parents_in_tree(root)
        node = NodePereebator(attribute_names, function_names).visit(root)
    except Exception as ex:
        print(ex)
        return None
    return node, (attribute_names, function_names)


def replace_names(root, attribute_names, function_names):
    return NodePerehuyator(attribute_names, function_names).visit(root)


def get_most_common_names(all_names):
    common_names = defaultdict(int)
    for name in all_names:
        common_names[name] += 1
    common_names = list(common_names.items())
    common_names.sort(key=lambda it: it[1], reverse=True)
    return {name for name, count in common_names[:100]}


def process_parsed(name):
    parsed = pd.read_csv(name)
    parsed = parsed[pd.notnull(parsed['question_text'])]
    parsed = parsed[pd.notnull(parsed['answer_text'])]
    parsed = parsed[pd.notnull(parsed['answer_code'])]
    inputs = [question + answer for question, answer in zip(parsed['question_text'], parsed['answer_text'])]
    code = [(ast.parse(it), it) for it in parsed['answer_code']]
    code_common_names = [extract_common_names(*tree) for tree in code]

    attribute_names, function_names = [], []
    for _, (attributes, functions) in code_common_names:
        attribute_names += attributes.values()
        function_names += functions.values()

    attribute_names = get_most_common_names(attribute_names)
    function_names = get_most_common_names(function_names)

    renamed_code = [
        (replace_names(tree, attribute_names, function_names), text)
        for tree, text in code
    ]

    exit(99)


if __name__ == '__main__':
    # parse('QueryResults.csv')
    process_parsed('ParsedData.csv')
