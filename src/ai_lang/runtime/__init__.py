import json
import time
from dataclasses import dataclass, field
from typing import Any
from ..compiler import ExecutionGraph, ExecNode
from ..ir import IROp, Constraint, ConstraintOp
from ..skills import registry as skill_registry
from .. import skills as _skills_pkg
from ..llm import BaseLLM, MockLLM

try:
    from ..skills import robotics as _robotics_skills  # noqa: F401
except ImportError:
    pass


@dataclass
class RuntimeResult:
    task_name: str
    success: bool
    outputs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    events: list = field(default_factory=list)
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    constraint_violations: list = field(default_factory=list)
    repair_attempts: int = 0
    budget_exceeded: bool = False
    errors: list = field(default_factory=list)


class AIRuntime:
    def __init__(
        self,
        llm: BaseLLM = None,
        use_llm: bool = False,
        max_retries: int = 3,
        max_tokens: int = None,
        max_cost: float = None,
    ):
        self.llm = llm or MockLLM()
        self.use_llm = use_llm
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self._state: dict[str, Any] = {}
        self._tokens_used: int = 0
        self._cost_used: float = 0.0

    def execute(self, graph: ExecutionGraph, inputs: dict = None) -> RuntimeResult:
        result = RuntimeResult(task_name=graph.task_name, success=True)
        self._state = dict(inputs or {})
        self._tokens_used = 0
        self._cost_used = 0.0

        self._emit_event(result, "task_start", {"task": graph.task_name, "inputs": list(self._state.keys())})
        result.log.append(f"Starting task: {graph.task_name}")
        result.log.append(f"Inputs: {list(self._state.keys())}")

        execution_levels = graph.get_execution_order()
        result.log.append(f"Execution levels: {len(execution_levels)}")

        for level_idx, level in enumerate(execution_levels):
            if self._budget_exceeded(result):
                break
            result.log.append(f"--- Level {level_idx} ---")
            for node in level:
                if self._budget_exceeded(result):
                    break
                self._execute_node(node, result)

        if result.budget_exceeded:
            result.success = False
            result.log.append("STOPPED: budget exceeded")

        self._collect_outputs(graph, result)
        self._run_verification(graph, result)

        if result.constraint_violations and not result.budget_exceeded:
            result.success = False
            result.log.append(f"FAILED: {len(result.constraint_violations)} constraint violations")

        result.total_llm_calls = getattr(self.llm, 'call_count', 0)
        result.total_llm_tokens = self._tokens_used
        result.total_llm_cost = self._cost_used
        result.metrics = {
            "llm_calls": result.total_llm_calls,
            "tokens_used": self._tokens_used,
            "cost_usd": round(self._cost_used, 6),
            "repair_attempts": result.repair_attempts,
            "budget_exceeded": result.budget_exceeded,
        }

        self._emit_event(result, "task_end", {
            "success": result.success,
            "tokens": self._tokens_used,
            "cost": round(self._cost_used, 6),
        })

        return result

    def _execute_node(self, node: ExecNode, result: RuntimeResult):
        skill = skill_registry.get(node.func)
        if not skill:
            result.errors.append(f"Unknown skill: {node.func}")
            result.log.append(f"  [{node.id}] ERROR: no skill '{node.func}'")
            self._emit_event(result, "skill_error", {"skill": node.func, "error": "not found"})
            return

        resolved_args, resolved_kwargs = self._resolve_args(node)

        if self.use_llm and skill.uses_llm:
            self._execute_with_llm(node, skill, resolved_args, resolved_kwargs, result)
        else:
            result.log.append(f"  [{node.id}] EXEC: {node.func}({resolved_args}, {resolved_kwargs})")
            output = skill.execute(*resolved_args, **resolved_kwargs)
            self._state[node.result_key] = output
            result.state[node.result_key] = output
            result.log.append(f"  [{node.id}] -> {node.result_key} = {str(output)[:80]}")
            self._emit_event(result, "skill_exec", {"skill": node.func, "output_type": type(output).__name__})

    def _execute_with_llm(self, node: ExecNode, skill, resolved_args: list, resolved_kwargs: dict, result: RuntimeResult):
        prompt = skill.build_prompt(**resolved_kwargs, **{f"arg{i}": a for i, a in enumerate(resolved_args)})
        output = None

        for attempt in range(1, self.max_retries + 1):
            if not self._check_budget(result):
                return

            result.log.append(f"  [{node.id}] LLM call: {node.func} (attempt {attempt})")
            self._emit_event(result, "llm_call", {"skill": node.func, "attempt": attempt})

            try:
                llm_response = self.llm.call(prompt, context=dict(self._state))
            except Exception as e:
                result.log.append(f"  [{node.id}] LLM error: {e}")
                self._emit_event(result, "llm_error", {"skill": node.func, "error": str(e)})
                if attempt < self.max_retries:
                    continue
                result.errors.append(f"LLM call failed after {self.max_retries} attempts: {e}")
                return

            self._tokens_used += llm_response.tokens_used
            self._cost_used += llm_response.cost

            self._emit_event(result, "llm_response", {
                "skill": node.func,
                "attempt": attempt,
                "tokens": llm_response.tokens_used,
                "cost": round(llm_response.cost, 6),
            })

            try:
                output = llm_response.as_json()
            except Exception:
                output = {"raw": llm_response.content}

            if skill.output_schema:
                valid, error = skill.validate_output(output)
                if valid:
                    break
                result.log.append(f"  [{node.id}] Schema violation: {error}")
                self._emit_event(result, "schema_violation", {"skill": node.func, "error": error, "attempt": attempt})
                result.repair_attempts += 1
                prompt = skill.build_repair_prompt(
                    original_prompt=skill.build_prompt(**resolved_kwargs),
                    invalid_output=json.dumps(output) if isinstance(output, dict) else str(output),
                    error=error,
                )
                if attempt == self.max_retries:
                    result.errors.append(f"Schema validation failed after {self.max_retries} attempts: {error}")
            else:
                break

        self._state[node.result_key] = output
        result.state[node.result_key] = output
        result.log.append(f"  [{node.id}] -> {node.result_key} = {str(output)[:80]}")

    def _resolve_args(self, node: ExecNode) -> tuple[list, dict]:
        args = []
        for a in node.args:
            if isinstance(a, str) and a.startswith("${") and a.endswith("}"):
                key = a[2:-1]
                args.append(self._state.get(key))
            else:
                args.append(a)

        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                key = v[2:-1]
                kwargs[k] = self._state.get(key)
            else:
                kwargs[k] = v

        return args, kwargs

    def _check_budget(self, result: RuntimeResult) -> bool:
        if self.max_tokens and self._tokens_used >= self.max_tokens:
            result.budget_exceeded = True
            result.log.append(f"  BUDGET EXCEEDED: tokens {self._tokens_used} >= {self.max_tokens}")
            self._emit_event(result, "budget_exceeded", {"type": "tokens", "used": self._tokens_used, "limit": self.max_tokens})
            return False
        if self.max_cost and self._cost_used >= self.max_cost:
            result.budget_exceeded = True
            result.log.append(f"  BUDGET EXCEEDED: cost ${self._cost_used:.6f} >= ${self.max_cost:.6f}")
            self._emit_event(result, "budget_exceeded", {"type": "cost", "used": self._cost_used, "limit": self.max_cost})
            return False
        return True

    def _budget_exceeded(self, result: RuntimeResult) -> bool:
        return result.budget_exceeded

    def _resolve_metric(self, metric: str) -> Any:
        if metric in self._state:
            val = self._state[metric]
            if isinstance(val, dict):
                return val.get(metric)
            return val
        for key, val in self._state.items():
            if isinstance(val, dict) and metric in val:
                return val[metric]
        return None

    def _collect_outputs(self, graph: ExecutionGraph, result: RuntimeResult):
        all_constraints_passed = len(result.constraint_violations) == 0
        for name, type_name in graph.outputs:
            value = None
            if name in self._state:
                value = self._state[name]
            elif name in result.state:
                value = result.state[name]
            else:
                result.errors.append(f"Output '{name}' not produced")
                continue

            result.outputs[name] = {
                "value": value,
                "type": type_name,
                "verified": all_constraints_passed,
                "trust_level": "verified" if all_constraints_passed else "unverified",
            }

    def _run_verification(self, graph: ExecutionGraph, result: RuntimeResult):
        for constraint in graph.constraints:
            self._check_constraint(constraint, result)

        for node in graph.nodes:
            if node.op == IROp.VERIFY:
                for check in node.checks:
                    passed = len(result.constraint_violations) == 0
                    result.log.append(f"  VERIFY: {check} -> {'PASS' if passed else 'FAIL'}")
                    self._emit_event(result, "verify_check", {"check": check, "passed": passed})

    def _check_constraint(self, constraint: Constraint, result: RuntimeResult):
        actual = self._resolve_metric(constraint.metric)
        if actual is None:
            result.constraint_violations.append(
                f"Metric '{constraint.metric}' not available for constraint check"
            )
            self._emit_event(result, "constraint_check", {"metric": constraint.metric, "passed": False, "reason": "metric not found"})
            return

        if not isinstance(actual, (int, float)):
            result.constraint_violations.append(
                f"Metric '{constraint.metric}' is not numeric: {type(actual).__name__}"
            )
            self._emit_event(result, "constraint_check", {"metric": constraint.metric, "passed": False, "reason": "not numeric"})
            return

        violated = False
        if constraint.op == ConstraintOp.LT:
            violated = not (actual < constraint.value)
        elif constraint.op == ConstraintOp.GT:
            violated = not (actual > constraint.value)
        elif constraint.op == ConstraintOp.LE:
            violated = not (actual <= constraint.value)
        elif constraint.op == ConstraintOp.GE:
            violated = not (actual >= constraint.value)
        elif constraint.op == ConstraintOp.EQ:
            violated = not (actual == constraint.value)
        elif constraint.op == ConstraintOp.NE:
            violated = not (actual != constraint.value)

        if violated:
            msg = f"{constraint.metric} {constraint.op.value} {constraint.value}: got {actual}"
            result.constraint_violations.append(msg)
            result.log.append(f"  CONSTRAINT FAIL: {msg}")
            self._emit_event(result, "constraint_check", {"metric": constraint.metric, "passed": False, "value": actual})
        else:
            result.log.append(f"  CONSTRAINT OK: {constraint.metric}={actual} {constraint.op.value} {constraint.value}")
            self._emit_event(result, "constraint_check", {"metric": constraint.metric, "passed": True, "value": actual})

    def _emit_event(self, result: RuntimeResult, event_type: str, data: dict):
        result.events.append({
            "type": event_type,
            "timestamp": time.time(),
            **data,
        })


def run(
    graph: ExecutionGraph,
    inputs: dict = None,
    llm: BaseLLM = None,
    use_llm: bool = False,
    max_retries: int = 3,
    max_tokens: int = None,
    max_cost: float = None,
) -> RuntimeResult:
    runtime = AIRuntime(llm=llm, use_llm=use_llm, max_retries=max_retries, max_tokens=max_tokens, max_cost=max_cost)
    return runtime.execute(graph, inputs)
