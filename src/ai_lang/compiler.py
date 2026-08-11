from dataclasses import dataclass, field
from typing import Any, Optional
from .ir import TaskIR, Block, IROp, Assignment, Objective, Constraint, VerifyCheck


@dataclass
class ExecNode:
    id: str
    op: IROp
    func: Optional[str] = None
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)
    result_key: Optional[str] = None
    objectives: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutionGraph:
    task_name: str
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    constraints: list = field(default_factory=list)

    def add_node(self, node: ExecNode):
        self.nodes.append(node)

    def get_execution_order(self) -> list[list[ExecNode]]:
        executed = set()
        remaining = list(self.nodes)
        levels = []

        while remaining:
            ready = []
            not_ready = []
            for node in remaining:
                if all(dep in executed for dep in node.deps):
                    ready.append(node)
                else:
                    not_ready.append(node)
            if not ready:
                raise RuntimeError(f"Circular dependency detected among: {[n.id for n in not_ready]}")
            levels.append(ready)
            for n in ready:
                executed.add(n.id)
            remaining = not_ready

        return levels


class Compiler:
    def __init__(self, task: TaskIR):
        self.task = task
        self.graph = ExecutionGraph(task_name=task.name)
        self._node_counter = 0
        self._result_map: dict[str, str] = {}

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def compile(self) -> ExecutionGraph:
        self.graph.inputs = [(i.name, i.type_name) for i in self.task.inputs]
        self.graph.outputs = [(o.name, o.type_name) for o in self.task.outputs]

        for block in self.task.blocks:
            if block.op == IROp.PLAN:
                self.graph.objectives = block.objectives
                self.graph.constraints = block.constraints
            self._compile_block(block)

        return self.graph

    def _compile_block(self, block: Block):
        for stmt in block.statements:
            self._compile_statement(stmt, block)

    def _compile_statement(self, stmt: Assignment, block: Block):
        deps = []
        resolved_args = []
        for arg in stmt.args:
            if isinstance(arg, str) and arg in self._result_map:
                deps.append(self._result_map[arg])
                resolved_args.append(f"${{{arg}}}")
            else:
                resolved_args.append(arg)

        resolved_kwargs = {}
        for k, v in stmt.kwargs.items():
            if isinstance(v, str) and v in self._result_map:
                deps.append(self._result_map[v])
                resolved_kwargs[k] = f"${{{v}}}"
            else:
                resolved_kwargs[k] = v

        node = ExecNode(
            id=self._next_id(stmt.target),
            op=block.op,
            func=stmt.func,
            args=resolved_args,
            kwargs=resolved_kwargs,
            deps=deps,
            result_key=stmt.target,
            objectives=block.objectives if block.op == IROp.PLAN else [],
            constraints=block.constraints if block.op == IROp.PLAN else [],
            checks=[c.name for c in block.checks] if block.op == IROp.VERIFY else [],
        )

        self.graph.add_node(node)
        self._result_map[stmt.target] = node.id


def compile_task(task: TaskIR) -> ExecutionGraph:
    return Compiler(task).compile()
