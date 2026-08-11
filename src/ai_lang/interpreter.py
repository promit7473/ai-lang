"""AI Lang Interpreter.

Executes AI Lang programs end-to-end:
- Parses .ais source
- Builds IR
- Executes COMPUTE/EVALUATE blocks with expression engine
- Manages skill calls, variables, control flow
"""
import re
from typing import Any
from .ir import TaskIR, Block, IROp
from .expr import (
    ExprContext, parse_expr, eval_expr, eval_text,
    format_value, interpolate_string,
)
from .skills import registry as skill_registry


class InterpreterError(Exception):
    pass


class AIInterpreter:
    def __init__(self, extra_skills: dict = None):
        self.ctx = ExprContext(variables={}, skills={}, log=[])
        self._register_builtin_skills()
        if extra_skills:
            for name, func in extra_skills.items():
                self.ctx.skills[name] = func

    def _register_builtin_skills(self):
        for skill in skill_registry.all():
            self.ctx.skills[skill.name] = skill.execute
            self.ctx.skills[skill.func] = skill.execute

    def execute(self, task: TaskIR, inputs: dict = None) -> dict:
        for inp in task.inputs:
            if inp.description:
                try:
                    value = int(inp.description)
                except ValueError:
                    try:
                        value = float(inp.description)
                    except ValueError:
                        value = inp.description
                self.ctx.set_var(inp.name, value)
        if inputs:
            for k, v in inputs.items():
                self.ctx.set_var(k, v)

        result = {
            "task_name": task.name,
            "success": True,
            "outputs": {},
            "log": [],
            "variables": {},
        }

        for block in task.blocks:
            try:
                if block.op == IROp.COMPUTE:
                    self._execute_compute_block(block, result)
                elif block.op == IROp.EVALUATE:
                    self._execute_compute_block(block, result)
                elif block.op == IROp.VERIFY:
                    self._execute_verify_block(block, result)
                elif block.op == IROp.OUTPUT:
                    self._execute_output_declaration(block, task, result)
                elif block.op in (IROp.PERCEIVE, IROp.REASON, IROp.PLAN, IROp.CONTROL):
                    self._execute_skill_block(block, result)
            except Exception as e:
                result["success"] = False
                result["log"].append(f"ERROR in {block.op.value}: {str(e)}")
                raise InterpreterError(f"Block {block.op.value} failed: {e}") from e

        result["variables"] = dict(self.ctx.variables)
        return result

    def _execute_compute_block(self, block: Block, result: dict):
        stmts = block.statements
        i = 0
        while i < len(stmts):
            stmt = stmts[i].strip()
            if not stmt or stmt.startswith("#"):
                i += 1
                continue

            if stmt.startswith("FOR "):
                i = self._execute_for(stmts, i, result)
            elif stmt.startswith("IF "):
                i = self._execute_if(stmts, i, result)
            else:
                self._execute_statement(stmt, result)
                i += 1

    def _execute_statement(self, stmt: str, result: dict):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("#"):
            return

        if stmt.startswith("LOG "):
            self._execute_log(stmt, result)
        elif "=" in stmt and not stmt.startswith("("):
            self._execute_assignment(stmt, result)
        else:
            try:
                eval_text(stmt, self.ctx)
            except Exception as e:
                result["log"].append(f"  WARN: expression '{stmt[:50]}' failed: {e}")

    def _execute_for(self, stmts: list[str], start_idx: int, result: dict) -> int:
        stmt = stmts[start_idx]
        match = re.match(r"FOR\s+(\w+)\s+IN\s+(.+?)\s*\.\.\s*(.+?)\s*\{", stmt)
        if not match:
            raise SyntaxError(f"Invalid FOR statement: {stmt[:80]}")

        var_name = match.group(1)
        start_expr = match.group(2).strip()
        end_expr = match.group(3).strip()

        try:
            start = int(eval_text(start_expr, self.ctx))
        except (ValueError, TypeError):
            start = int(start_expr)
        try:
            end = int(eval_text(end_expr, self.ctx))
        except (ValueError, TypeError):
            end = int(end_expr)

        body_stmts = []
        i = start_idx + 1
        depth = 1
        while i < len(stmts) and depth > 0:
            line = stmts[i]
            opens = line.count("{")
            closes = line.count("}")
            depth += opens - closes
            if depth > 0:
                body_stmts.append(line)
            i += 1

        for val in range(start, end + 1):
            self.ctx.set_var(var_name, val)
            j = 0
            while j < len(body_stmts):
                s = body_stmts[j].strip()
                if s.startswith("FOR "):
                    j += 1
                    nested_body = []
                    nested_depth = 1
                    while j < len(body_stmts) and nested_depth > 0:
                        bl = body_stmts[j]
                        nested_depth += bl.count("{") - bl.count("}")
                        if nested_depth > 0:
                            nested_body.append(bl)
                        j += 1
                elif s.startswith("IF "):
                    j = self._execute_if(body_stmts, j, result)
                else:
                    self._execute_statement(s, result)
                    j += 1

        return i

    def _execute_if(self, stmts: list[str], start_idx: int, result: dict) -> int:
        stmt = stmts[start_idx]
        match = re.match(r"IF\s+(.+?)\s*\{", stmt)
        if not match:
            raise SyntaxError(f"Invalid IF statement: {stmt[:80]}")

        condition = match.group(1).strip()

        true_body = []
        false_body = []
        i = start_idx + 1
        depth = 1
        in_else = False
        while i < len(stmts) and depth > 0:
            line = stmts[i]
            opens = line.count("{")
            closes = line.count("}")
            if "ELSE" in line and depth == 1:
                in_else = True
                i += 1
                continue
            depth += opens - closes
            if depth > 0:
                if in_else:
                    false_body.append(line)
                else:
                    true_body.append(line)
            i += 1

        try:
            cond_result = eval_text(condition, self.ctx)
        except Exception as e:
            raise InterpreterError(f"IF condition failed: {condition} -> {e}")

        body = true_body if cond_result else false_body
        for body_stmt in body:
            self._execute_statement(body_stmt, result)

        return i

    def _execute_log(self, stmt: str, result: dict):
        match = re.match(r'LOG\s+"(.+)"', stmt)
        if not match:
            match = re.match(r"LOG\s+'(.+)'", stmt)
        if not match:
            raise SyntaxError(f"Invalid LOG statement: {stmt[:80]}")

        template = match.group(1)
        message = interpolate_string(template, self.ctx)
        result["log"].append(f"  {message}")
        self.ctx.log.append(message)

    def _execute_assignment(self, stmt: str, result: dict):
        eq_idx = stmt.index("=")
        target = stmt[:eq_idx].strip()
        expr_text = stmt[eq_idx + 1:].strip()

        value = eval_text(expr_text, self.ctx)
        self.ctx.set_var(target, value)

    def _split_statements(self, body: str) -> list[str]:
        statements = []
        current = []
        depth = 0

        for char in body:
            if char == "{":
                depth += 1
                current.append(char)
            elif char == "}":
                depth -= 1
                current.append(char)
            elif char == ";" and depth == 0:
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
            else:
                current.append(char)

        remaining = "".join(current).strip()
        if remaining:
            statements.append(remaining)

        return [s for s in statements if s]

    def _execute_skill_block(self, block: Block, result: dict):
        for stmt in block.statements:
            if isinstance(stmt, str):
                self._execute_statement(stmt, result)

    def _execute_verify_block(self, block: Block, result: dict):
        for check in block.checks:
            condition = check.condition or check.name
            try:
                check_result = eval_text(condition, self.ctx)
                if check_result:
                    result["log"].append(f"  CHECK PASS: {condition}")
                else:
                    result["log"].append(f"  CHECK FAIL: {condition}")
                    result["success"] = False
            except Exception as e:
                result["log"].append(f"  CHECK ERROR: {condition} -> {e}")
                result["success"] = False

    def _execute_output_declaration(self, block: Block, task: TaskIR, result: dict):
        for output_decl in task.outputs:
            name = output_decl.name
            if name in self.ctx.variables:
                result["outputs"][name] = self.ctx.variables[name]
            else:
                result["log"].append(f"  WARN: output '{name}' not found in variables")


def execute(source: str, inputs: dict = None, extra_skills: dict = None) -> dict:
    from .parser import parse
    task = parse(source)
    interpreter = AIInterpreter(extra_skills=extra_skills)
    return interpreter.execute(task, inputs)
