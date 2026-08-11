# Compiler API

## `compile_task(task: TaskIR) -> ExecutionGraph`

Compile an AI Lang IR into an execution graph.

```python
from ai_lang import parse, compile_task

task = parse(source)
graph = compile_task(task)

# Access nodes
for node in graph.nodes:
    print(f"{node.id}: {node.func}")

# Get topological execution order
levels = graph.get_execution_order()
for i, level in enumerate(levels):
    print(f"Level {i}: {[n.func for n in level]}")
```

## `ExecutionGraph`

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_name` | str | Task name |
| `inputs` | list | Input declarations |
| `outputs` | list | Output declarations |
| `nodes` | list[ExecNode] | Execution nodes |
| `objectives` | list | Optimization objectives |
| `constraints` | list | Verification constraints |

## `ExecNode`

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique node ID |
| `op` | IROp | Operation type |
| `func` | str | Skill function name |
| `args` | list | Positional arguments |
| `kwargs` | dict | Keyword arguments |
| `deps` | list | Node dependencies |
| `result_key` | str | Variable name for result |
