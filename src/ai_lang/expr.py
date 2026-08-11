"""AI Lang Expression Engine and Interpreter.

Adds computational capability to the language:
- Arithmetic: +, -, *, /, %
- Comparisons: <, >, <=, >=, ==, !=
- Logical: AND, OR, NOT
- Function calls: skill_name(args)
- Variable references: $var or var
- FOR loops: FOR var IN start..end { ... }
- IF statements: IF condition { ... } ELSE { ... }
- LOG: LOG "message {var}"
- Assignment: var = expr
"""
import re
import json
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class ExprContext:
    variables: dict = field(default_factory=dict)
    skills: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def get_var(self, name: str) -> Any:
        if name.startswith("$"):
            name = name[1:]
        if name not in self.variables:
            raise NameError(f"Variable '{name}' not defined")
        return self.variables[name]

    def set_var(self, name: str, value: Any):
        self.variables[name] = value

    def call_skill(self, name: str, args: list) -> Any:
        if name not in self.skills:
            raise NameError(f"Skill '{name}' not found")
        func = self.skills[name]
        return func(*args)


class Expr:
    pass


@dataclass
class Literal(Expr):
    value: Any


@dataclass
class Variable(Expr):
    name: str


@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class FunctionCall(Expr):
    name: str
    args: list[Expr]


@dataclass
class ArrayLiteral(Expr):
    elements: list[Expr]


@dataclass
class FieldAccess(Expr):
    obj: Expr
    field: str


@dataclass
class ArrayAccess(Expr):
    obj: Expr
    index: Expr


class ExprParser:
    TOKEN_RE = re.compile(r"""
        (?P<FLOAT>\d+\.\d+) |
        (?P<INT>\d+) |
        (?P<STRING>"[^"]*"|'[^']*') |
        (?P<CMP><=|>=|!=|<|>|==) |
        (?P<DOT>\.) |
        (?P<OP>[+\-*/%]) |
        (?P<LPAREN>\() |
        (?P<RPAREN>\)) |
        (?P<COMMA>,) |
        (?P<LBRACKET>\[) |
        (?P<RBRACKET>\]) |
        (?P<LBRACE>\{) |
        (?P<RBRACE>\}) |
        (?P<RANGE>\.\.) |
        (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*) |
        (?P<WS>\s+)
    """, re.VERBOSE)

    COMPARISONS = {"<", ">", "<=", ">=", "==", "!="}
    LOGICAL = {"AND", "OR"}

    def __init__(self, text: str):
        self.text = text
        self.tokens = self._tokenize()
        self.pos = 0

    def _tokenize(self) -> list[tuple[str, str]]:
        tokens = []
        pos = 0
        while pos < len(self.text):
            match = self.TOKEN_RE.match(self.text, pos)
            if not match:
                raise SyntaxError(f"Unexpected char '{self.text[pos]}' at position {pos}")
            kind = match.lastgroup
            value = match.group()
            if kind != "WS":
                tokens.append((kind, value))
            pos = match.end()
        return tokens

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, expected_value: str) -> str:
        tok = self.advance()
        if tok[1] != expected_value:
            raise SyntaxError(f"Expected '{expected_value}', got '{tok[1]}'")
        return tok[1]

    def parse(self) -> Expr:
        return self.parse_logical()

    def parse_logical(self) -> Expr:
        left = self.parse_comparison()
        while self.peek() and self.peek()[0] == "IDENT" and self.peek()[1] in ("AND", "OR"):
            op = self.advance()[1]
            right = self.parse_comparison()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_comparison(self) -> Expr:
        left = self.parse_additive()
        while self.peek() and self.peek()[0] == "CMP":
            op = self.advance()[1]
            right = self.parse_additive()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_additive(self) -> Expr:
        left = self.parse_multiplicative()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] in ("+", "-"):
            op = self.advance()[1]
            right = self.parse_multiplicative()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_multiplicative(self) -> Expr:
        left = self.parse_unary()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] in ("*", "/", "%"):
            op = self.advance()[1]
            right = self.parse_unary()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_unary(self) -> Expr:
        if self.peek() and self.peek()[0] == "OP" and self.peek()[1] == "-":
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op="-", operand=operand)
        if self.peek() and self.peek()[0] == "IDENT" and self.peek()[1] == "NOT":
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op="NOT", operand=operand)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if not tok:
            raise SyntaxError("Unexpected end of expression")

        if tok[0] == "FLOAT":
            self.advance()
            return Literal(value=float(tok[1]))

        if tok[0] == "INT":
            self.advance()
            return Literal(value=int(tok[1]))

        if tok[0] == "STRING":
            self.advance()
            val = tok[1]
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            return Literal(value=val)

        if tok[0] == "IDENT":
            name = self.advance()[1]
            node: Expr = Variable(name=name)
            while self.peek() and self.peek()[0] in ("DOT", "LBRACKET"):
                if self.peek()[0] == "DOT":
                    self.advance()
                    field_tok = self.advance()
                    if field_tok[0] != "IDENT":
                        raise SyntaxError(f"Expected field name after '.', got '{field_tok[1]}'")
                    node = FieldAccess(obj=node, field=field_tok[1])
                elif self.peek()[0] == "LBRACKET":
                    self.advance()
                    idx = self.parse_logical()
                    if self.peek() and self.peek()[0] == "RBRACKET":
                        self.advance()
                    node = ArrayAccess(obj=node, index=idx)
            if isinstance(node, Variable) and self.peek() and self.peek()[0] == "LPAREN":
                self.advance()
                args = []
                if self.peek() and self.peek()[0] != "RPAREN":
                    args.append(self.parse_logical())
                    while self.peek() and self.peek()[0] == "COMMA":
                        self.advance()
                        args.append(self.parse_logical())
                if self.peek() and self.peek()[0] == "RPAREN":
                    self.advance()
                return FunctionCall(name=node.name, args=args)
            return node

        if tok[0] == "LPAREN":
            self.advance()
            expr = self.parse_logical()
            if self.peek() and self.peek()[0] == "RPAREN":
                self.advance()
            return expr

        if tok[0] == "LBRACKET":
            self.advance()
            elements = []
            if self.peek() and self.peek()[0] != "RBRACKET":
                elements.append(self.parse_logical())
                while self.peek() and self.peek()[0] == "COMMA":
                    self.advance()
                    elements.append(self.parse_logical())
            if self.peek() and self.peek()[0] == "RBRACKET":
                self.advance()
            return ArrayLiteral(elements=elements)

        raise SyntaxError(f"Unexpected token: {tok[1]}")


def parse_expr(text: str) -> Expr:
    parser = ExprParser(text)
    return parser.parse()


def eval_expr(expr: Expr, ctx: ExprContext) -> Any:
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, Variable):
        return ctx.get_var(expr.name)

    if isinstance(expr, BinaryOp):
        left = eval_expr(expr.left, ctx)
        right = eval_expr(expr.right, ctx)
        if expr.op == "+":
            return left + right
        elif expr.op == "-":
            return left - right
        elif expr.op == "*":
            return left * right
        elif expr.op == "/":
            return left / right if right != 0 else 0
        elif expr.op == "%":
            return left % right
        elif expr.op == "<":
            return left < right
        elif expr.op == ">":
            return left > right
        elif expr.op == "<=":
            return left <= right
        elif expr.op == ">=":
            return left >= right
        elif expr.op == "==":
            return left == right
        elif expr.op == "!=":
            return left != right
        elif expr.op == "AND":
            return left and right
        elif expr.op == "OR":
            return left or right

    if isinstance(expr, UnaryOp):
        operand = eval_expr(expr.operand, ctx)
        if expr.op == "-":
            return -operand
        elif expr.op == "NOT":
            return not operand

    if isinstance(expr, FunctionCall):
        args = [eval_expr(a, ctx) for a in expr.args]
        return ctx.call_skill(expr.name, args)

    if isinstance(expr, ArrayLiteral):
        return [eval_expr(e, ctx) for e in expr.elements]

    if isinstance(expr, FieldAccess):
        obj = eval_expr(expr.obj, ctx)
        if isinstance(obj, dict):
            return obj.get(expr.field)
        return getattr(obj, expr.field, None)

    if isinstance(expr, ArrayAccess):
        obj = eval_expr(expr.obj, ctx)
        idx = eval_expr(expr.index, ctx)
        if isinstance(obj, (list, tuple)):
            return obj[int(idx)] if int(idx) < len(obj) else None
        if isinstance(obj, dict):
            return obj.get(idx)
        return None

    raise TypeError(f"Unknown expression type: {type(expr)}")


def eval_text(text: str, ctx: ExprContext) -> Any:
    expr = parse_expr(text.strip())
    return eval_expr(expr, ctx)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    return str(value)


def interpolate_string(text: str, ctx: ExprContext) -> str:
    def replacer(match):
        expr_text = match.group(1)
        try:
            result = eval_text(expr_text, ctx)
            return format_value(result)
        except Exception:
            return match.group(0)
    return re.sub(r"\{([^}]+)\}", replacer, text)
