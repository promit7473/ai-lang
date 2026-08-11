import re
from typing import Any
from .ir import (
    TaskIR, Block, InputDecl, Assignment, Objective, Constraint,
    VerifyCheck, OutputDecl, IROp, ObjectiveType, ConstraintOp
)


class Token:
    def __init__(self, type_: str, value: str, line: int):
        self.type = type_
        self.value = value
        self.line = line

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, L{self.line})"


class Lexer:
    KEYWORDS = {
        'TASK', 'INPUT', 'PERCEIVE', 'REASON', 'PLAN', 'CONTROL',
        'VERIFY', 'OUTPUT', 'OBJECTIVE', 'CONSTRAINT', 'CHECK',
        'MAXIMIZE', 'MINIMIZE', 'TARGET',
        'COMPUTE', 'EVALUATE',
        'FOR', 'IN', 'IF', 'ELSE', 'LOG',
        'AND', 'OR', 'NOT',
    }

    TOKEN_SPEC = [
        ('SKIP',      r'[ \t]+'),
        ('NEWLINE',   r'\n'),
        ('COMMENT',   r'#[^\n]*'),
        ('ARROW',     r'->'),
        ('LBRACE',    r'\{'),
        ('RBRACE',    r'\}'),
        ('LPAREN',    r'\('),
        ('RPAREN',    r'\)'),
        ('COMMA',     r','),
        ('COLON',     r':'),
        ('SEMI',      r';'),
        ('CMP',       r'<=|>=|!=|==|<|>'),
        ('EQUALS',    r'='),
        ('RANGE',     r'\.\.'),
        ('DOT',       r'\.'),
        ('FLOAT',     r'\d+\.\d+'),
        ('INT',       r'\d+'),
        ('OP',        r'[+\-*/%]'),
        ('IDENT',     r'[A-Za-z_][A-Za-z0-9_]*'),
        ('STRING',    r'"[^"]*"'),
    ]

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.tokens: list[Token] = []
        self._tokenize()

    def _tokenize(self):
        pattern = '|'.join(f'(?P<{name}>{regex})' for name, regex in self.TOKEN_SPEC)
        regex = re.compile(pattern)
        pos = 0
        while pos < len(self.source):
            match = regex.match(self.source, pos)
            if not match:
                raise SyntaxError(f"Unexpected char {self.source[pos]!r} at line {self.line}")
            kind = match.lastgroup
            value = match.group()
            if kind == 'NEWLINE':
                self.tokens.append(Token('NEWLINE', value, self.line))
                self.line += 1
            elif kind in ('SKIP', 'COMMENT'):
                pass
            elif kind == 'IDENT' and value in self.KEYWORDS:
                self.tokens.append(Token(value, value, self.line))
            elif kind == 'IDENT' and value.upper() in ('MINIMIZE', 'MAXIMIZE', 'TARGET'):
                self.tokens.append(Token(value.upper(), value, self.line))
            elif kind == 'STRING':
                self.tokens.append(Token('STRING', value, self.line))
            elif kind == 'INT':
                self.tokens.append(Token('INT', value, self.line))
            elif kind == 'FLOAT':
                self.tokens.append(Token('FLOAT', value, self.line))
            else:
                self.tokens.append(Token(kind, value, self.line))
            pos = match.end()
        self.tokens.append(Token('EOF', '', self.line))


class Parser:
    CMP_MAP = {
        '<': ConstraintOp.LT,
        '>': ConstraintOp.GT,
        '<=': ConstraintOp.LE,
        '>=': ConstraintOp.GE,
        '=': ConstraintOp.EQ,
        '!=': ConstraintOp.NE,
    }

    OBJ_MAP = {
        'MAXIMIZE': ObjectiveType.MAXIMIZE,
        'MINIMIZE': ObjectiveType.MINIMIZE,
        'TARGET': ObjectiveType.TARGET,
    }

    BLOCK_OPS = {
        'PERCEIVE': IROp.PERCEIVE,
        'REASON': IROp.REASON,
        'PLAN': IROp.PLAN,
        'CONTROL': IROp.CONTROL,
        'VERIFY': IROp.VERIFY,
        'COMPUTE': IROp.COMPUTE,
        'EVALUATE': IROp.EVALUATE,
    }

    def __init__(self, source: str):
        self.lexer = Lexer(source)
        self.tokens = self.lexer.tokens
        self.pos = 0

    def peek(self) -> Token:
        idx = self.pos
        while idx < len(self.tokens) and self.tokens[idx].type == 'NEWLINE':
            idx += 1
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self) -> Token:
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == 'NEWLINE':
            self.pos += 1
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: str, value: str = None) -> Token:
        tok = self.peek()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_} but got {tok.type}({tok.value!r}) at line {tok.line}")
        if value is not None and tok.value != value:
            raise SyntaxError(f"Expected {value!r} but got {tok.value!r} at line {tok.line}")
        return self.advance()

    def parse(self) -> TaskIR:
        self.expect('TASK')
        name_tok = self.expect('IDENT')
        self.expect('LBRACE')

        task = TaskIR(name=name_tok.value)

        while self.peek().type != 'RBRACE':
            tok = self.peek()
            if tok.type == 'INPUT':
                self._parse_input(task)
            elif tok.type == 'OUTPUT':
                self._parse_output(task)
            elif tok.type in self.BLOCK_OPS:
                self._parse_block(task)
            else:
                raise SyntaxError(f"Unexpected token {tok.type}({tok.value!r}) at line {tok.line}")

        self.expect('RBRACE')
        self.expect('EOF')
        return task

    def _parse_input(self, task: TaskIR):
        self.advance()
        self.expect('LBRACE')
        while self.peek().type != 'RBRACE':
            name = self.expect('IDENT').value
            if self.peek().type == 'COLON':
                self.advance()
                type_name = self.expect('IDENT').value
                task.inputs.append(InputDecl(name=name, type_name=type_name))
            elif self.peek().type == 'EQUALS':
                self.advance()
                default_tok = self.advance()
                if default_tok.type == 'INT':
                    task.inputs.append(InputDecl(name=name, type_name="int", description=str(int(default_tok.value))))
                elif default_tok.type == 'FLOAT':
                    task.inputs.append(InputDecl(name=name, type_name="float", description=str(float(default_tok.value))))
                elif default_tok.type == 'STRING':
                    task.inputs.append(InputDecl(name=name, type_name="string", description=default_tok.value))
                else:
                    task.inputs.append(InputDecl(name=name, type_name="unknown", description=default_tok.value))
            else:
                task.inputs.append(InputDecl(name=name, type_name="unknown"))
            if self.peek().type == 'COMMA':
                self.advance()
        self.expect('RBRACE')

    def _parse_output(self, task: TaskIR):
        self.advance()
        self.expect('LBRACE')
        while self.peek().type != 'RBRACE':
            name = self.expect('IDENT').value
            type_name = "any"
            if self.peek().type == 'COLON':
                self.advance()
                type_name = self.expect('IDENT').value
            task.outputs.append(OutputDecl(name=name, type_name=type_name))
            if self.peek().type == 'COMMA':
                self.advance()
        self.expect('RBRACE')

    def _parse_block(self, task: TaskIR):
        op_tok = self.advance()
        op = self.BLOCK_OPS[op_tok.type]
        block = Block(op=op)
        self.expect('LBRACE')

        if op in (IROp.COMPUTE, IROp.EVALUATE):
            block.statements = self._parse_compute_body()
        else:
            while self.peek().type != 'RBRACE':
                if self.peek().type == 'OBJECTIVE':
                    block.objectives.append(self._parse_objective())
                elif self.peek().type == 'CONSTRAINT':
                    block.constraints.append(self._parse_constraint())
                elif self.peek().type == 'CHECK':
                    block.checks.append(self._parse_check())
                else:
                    block.statements.append(self._parse_statement())

        self.expect('RBRACE')
        task.blocks.append(block)

    def _parse_compute_body(self) -> list[str]:
        statements = []
        current_tokens = []
        indent_level = 1

        while self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            if tok.type == 'RBRACE' and indent_level <= 1:
                if current_tokens:
                    stmt = " ".join(t.value for t in current_tokens).strip()
                    if stmt:
                        statements.append(stmt)
                break
            elif tok.type == 'LBRACE':
                indent_level += 1
                current_tokens.append(tok)
                self.pos += 1
            elif tok.type == 'RBRACE':
                indent_level -= 1
                if indent_level <= 0:
                    if current_tokens:
                        stmt = " ".join(t.value for t in current_tokens).strip()
                        if stmt:
                            statements.append(stmt)
                    break
                current_tokens.append(tok)
                self.pos += 1
            elif tok.type == 'NEWLINE':
                if current_tokens:
                    stmt = " ".join(t.value for t in current_tokens).strip()
                    if stmt:
                        statements.append(stmt)
                    current_tokens = []
                self.pos += 1
            elif tok.type in ('SKIP', 'COMMENT'):
                self.pos += 1
            else:
                current_tokens.append(tok)
                self.pos += 1

        return [s.strip() for s in statements if s.strip()]

    def _parse_objective(self) -> Objective:
        self.advance()
        obj_tok = self.advance()
        metric = self.expect('IDENT').value
        return Objective(type=self.OBJ_MAP[obj_tok.type], metric=metric)

    def _parse_constraint(self) -> Constraint:
        self.advance()
        metric = self.expect('IDENT').value
        cmp_tok = self.advance()
        value_tok = self.advance()
        value: Any
        if value_tok.type == 'INT':
            value = int(value_tok.value)
        elif value_tok.type == 'FLOAT':
            value = float(value_tok.value)
        elif value_tok.type == 'STRING':
            value = value_tok.value
        else:
            value = value_tok.value
        unit = ""
        if self.peek().type == 'IDENT':
            unit = self.advance().value
        return Constraint(metric=metric, op=self.CMP_MAP[cmp_tok.value], value=value, unit=unit)

    def _parse_check(self) -> VerifyCheck:
        self.advance()
        tokens = []
        while self.peek().type not in ('NEWLINE', 'RBRACE', 'EOF'):
            tokens.append(self.advance().value)
        condition = " ".join(tokens).strip()
        return VerifyCheck(name=condition, condition=condition)

    def _parse_statement(self) -> Assignment:
        target = self.expect('IDENT').value
        self.expect('EQUALS')
        func = self.expect('IDENT').value
        args, kwargs = self._parse_call_args()
        return Assignment(target=target, func=func, args=args, kwargs=kwargs)

    def _parse_call_args(self) -> tuple[list, dict]:
        args = []
        kwargs = {}
        if self.peek().type != 'LPAREN':
            return args, kwargs
        self.advance()
        while self.peek().type != 'RPAREN':
            if self.peek().type == 'IDENT' and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'EQUALS':
                key = self.advance().value
                self.advance()
                val = self._parse_value()
                kwargs[key] = val
            else:
                args.append(self._parse_value())
            if self.peek().type == 'COMMA':
                self.advance()
        self.expect('RPAREN')
        return args, kwargs

    def _parse_value(self) -> Any:
        tok = self.peek()
        if tok.type == 'INT':
            self.advance()
            return int(tok.value)
        elif tok.type == 'FLOAT':
            self.advance()
            return float(tok.value)
        elif tok.type == 'STRING':
            self.advance()
            return tok.value
        elif tok.type == 'IDENT':
            self.advance()
            return tok.value
        else:
            raise SyntaxError(f"Expected value but got {tok.type}({tok.value!r}) at line {tok.line}")


def parse(source: str) -> TaskIR:
    return Parser(source).parse()
