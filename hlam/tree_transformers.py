import ast

CONSTANT_LITERAL_TYPE = 'ConstantLiteral'
EMPTY_TOKEN = 'EmptyToken'
DICT_EXPAND_KEY = 'DICT_EXPAND_KEY'


def set_parents_in_tree(root):
    def set_parent(node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
        for child in ast.iter_child_nodes(node):
            set_parent(child)

    set_parent(root)


class NodeChecker(ast.NodeVisitor):
    def visit_BinOp(self, node):
        if not hasattr(node, 'left'):
            print('aaaa')


def make_constant(text):
    return {
        'type': CONSTANT_LITERAL_TYPE,
        'value': str(text),
    }


def make_empty_token():
    return {
        'type': EMPTY_TOKEN,
    }


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
        node = self.generic_visit(node)
        if node.id not in self.common_functions:
            if node.id not in self.function_replace:
                self.function_replace[node.id] = len(self.function_replace)
            node.id = 'f' + str(self.function_replace[node.id])
        return node

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)
        if node.attr not in self.common_attributes:
            if node.attr not in self.attribute_replace:
                self.attribute_replace[node.attr] = len(self.attribute_replace)
            node.attr = 'a' + str(self.attribute_replace[node.attr])
        return node


class NodeUebator(ast.NodeTransformer):
    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                value = self.visit(value)
            elif isinstance(value, list):
                value = [self.visit(it) if isinstance(it, ast.AST) else it for it in value]
            setattr(node, field, value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        node = self.generic_visit(node)
        return {
            'type': 'Assign',
            'targets': [node.target],
            'value': node.value,
        }

    def visit_Assert(self, node: ast.Assert):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'msg': node.msg,
        }

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'targets': node.targets,
            'value': node.value,
        }

    def visit_AsyncFor(self, node: ast.AsyncFor):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'target': node.target,
            'body': node.body,
            'item': node.item,
            'else': node.orelse,
        }

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'name': node.name,
            'args': node.args,
            'body': node.body,
            'decorator_list': node.decorator_list,
            'returns': node.returns,
        }

    def visit_AsyncWith(self, node: ast.AsyncWith):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'items': node.items,
            'body': node.body,
        }

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'attr': make_constant(node.attr),
        }

    def visit_AugAssign(self, node: ast.AugAssign):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'target': node.target,
            'value': node.value,
            'op': node.op,
        }

    def visit_AugLoad(self, node: ast.AugLoad):
        node = self.generic_visit(node)
        return node

    def visit_AugStore(self, node: ast.AugStore):
        node = self.generic_visit(node)
        return node

    def visit_Await(self, node: ast.Await):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        return {
            'type': node.op['type'],
            'left': node.left,
            'right': node.right,
        }

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
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
            'value': make_constant(node.s),
        }

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'func': node.func,
            'args': node.args,
            'keywords': node.keywords,
        }

    def visit_ClassDef(self, node: ast.ClassDef):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'name': make_constant(node.name),
            'bases': node.bases,
            'keywords': node.keywords,
            # 'starargs': node.starargs,
            # 'kwargs': node.kwargs,
            'body': node.body,
            'decorator_list': node.decorator_list,
        }

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'left': node.left,
            'ops': node.ops,
            'comparators': node.comparators,
        }

    def visit_Constant(self, node: ast.Constant):
        print(f"unexpected node {node}")
        node = self.generic_visit(node)
        return node

    def visit_Continue(self, node: ast.Continue):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Del(self, node: ast.Del):
        return {
            'type': node.__class__.__name__,
        }

    def visit_Delete(self, node: ast.Delete):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'targets': node.targets,
        }

    def visit_Dict(self, node: ast.Dict):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'values': node.values,
            'keys': [key or DICT_EXPAND_KEY for key in node.keys]
        }

    def visit_DictComp(self, node: ast.DictComp):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'key': node.key,
            'value': node.value,
            'generators': node.generators,
        }

    def visit_Ellipsis(self, node: ast.Ellipsis):
        return {
            'type': node.__class__.__name__,
        }

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'exception_type': node.type,
            'body': node.body,
            'name': make_constant(node.name),
        }

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Expression(self, node: ast.Expression):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'body': node.body,
        }

    def visit_ExtSlice(self, node: ast.ExtSlice):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'dims': node.dims,
        }

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'target': node.target,
            'iter': node.iter,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_FormattedValue(self, node: ast.FormattedValue):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'format': node.format_spec or {
                'type': 'Str',
                'value': '',
            }
        }

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'name': make_constant(node.name),
            'args': node.args,
            'body': node.body,
            'decorators': node.decorator_list,
        }

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'element': node.elt,
            'generators': node.generators,
        }

    def visit_Global(self, node: ast.Global):
        return {
            'type': node.__class__.__name__,
            'names': [make_constant(name) for name in node.names],
        }

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_IfExp(self, node: ast.IfExp):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_Import(self, node: ast.Import):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'names': node.names,
        }

    def visit_ImportFrom(self, node: ast.ImportFrom):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'names': node.names,
            'module': make_constant(node.module),
            'level': make_constant(node.level),
        }

    def visit_Index(self, node: ast.Index):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Interactive(self, node: ast.Interactive):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'values': node.values,
        }

    def visit_Lambda(self, node: ast.Lambda):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'args': node.args,
            'body': node.body,
        }

    def visit_List(self, node: ast.List):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_ListComp(self, node: ast.ListComp):
        node = self.generic_visit(node)
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
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'body': node.body,
        }

    def visit_Name(self, node: ast.Name):
        return {
            'type': node.__class__.__name__,
            'value': make_constant(node.id),
        }

    def visit_NameConstant(self, node: ast.NameConstant):
        return {
            'type': node.__class__.__name__,
            'value': make_constant(node.value),
        }

    def visit_Nonlocal(self, node: ast.Nonlocal):
        return {
            'type': node.__class__.__name__,
            'names': [make_constant(name) for name in node.names],
        }

    def visit_Num(self, node: ast.Num):
        return {
            'type': node.__class__.__name__,
            'value': make_constant(node.n),
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
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'exc': node.exc,
            'cause': node.cause,
        }

    def visit_Return(self, node: ast.Return):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_Set(self, node: ast.Set):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_SetComp(self, node: ast.SetComp):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'element': node.elt,
            'generators': node.generators,
        }

    def visit_Slice(self, node: ast.Slice):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'lower': node.lower,
            'upper': node.upper,
            'step': node.step,
        }

    def visit_Starred(self, node: ast.Starred):
        node = self.generic_visit(node)
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
            'value': make_constant(node.s),
        }

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'slice': node.slice,
            'operation': node.ctx,
        }

    def visit_Suite(self, node: ast.Suite):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_Try(self, node: ast.Try):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'body': node.body,
            'handlers': node.handlers,
            'else': node.orelse,
            'final': node.finalbody,
        }

    def visit_Tuple(self, node: ast.Tuple):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'elements': node.elts,
        }

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        return {
            'type': node.op['type'],
            'operand': node.operand,
        }

    def visit_While(self, node: ast.While):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'test': node.test,
            'then': node.body,
            'else': node.orelse,
        }

    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'items': node.items,
            'body': node.body,
        }

    def visit_Yield(self, node: ast.Yield):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_YieldFrom(self, node: ast.YieldFrom):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'value': node.value,
        }

    def visit_alias(self, node: ast.alias):
        return {
            'type': node.__class__.__name__,
            'name': make_constant(node.name),
            'asname': make_constant(node.asname),
        }

    def visit_arg(self, node: ast.arg):
        return {
            'type': node.__class__.__name__,
            'name': make_constant(node.arg),
        }

    def visit_arguments(self, node: ast.arguments):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'args': node.args,
            'vararg': node.vararg,
            'kwonlyargs': node.kwonlyargs,
            'kwarg': node.kwarg,
            'defaults': node.defaults,
            'kw_defaults': node.kw_defaults,
        }

    def visit_comprehension(self, node: ast.comprehension):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'iter': node.iter,
            'target': node.target,
            'ifs': node.ifs,
        }

    def visit_excepthandler(self, node: ast.excepthandler):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_expr(self, node: ast.expr):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_expr_context(self, node: ast.expr_context):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword):
        node = self.generic_visit(node)
        return {
            'type': node.__class__.__name__,
            'arg': make_constant(node.arg),
            'value': node.value,
        }

    def visit_mod(self, node: ast.mod):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_slice(self, node: ast.slice):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_stmt(self, node: ast.stmt):
        print(f'unexpected node {node}')
        node = self.generic_visit(node)
        return node

    def visit_withitem(self, node: ast.withitem):
        node = self.generic_visit(node)
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
