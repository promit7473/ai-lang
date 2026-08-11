"""End-to-end runner for the AI Lang vessel_monitor example."""
import sys
import json
sys.path.insert(0, "src")

from ai_lang import parse, compile_task, run


def main():
    with open("examples/vessel_monitor.ais", "r") as f:
        source = f.read()

    print("=" * 60)
    print("AI LANG — Vessel Monitor Example")
    print("=" * 60)

    print("\n[1] SOURCE CODE:")
    print("-" * 40)
    for i, line in enumerate(source.strip().split("\n"), 1):
        print(f"  {i:3}: {line}")

    print("\n[2] PARSING...")
    task = parse(source)
    print(f"  Task: {task.name}")
    print(f"  Inputs: {[(i.name, i.type_name) for i in task.inputs]}")
    print(f"  Blocks: {[(b.op.value, len(b.statements)) for b in task.blocks]}")

    print("\n[3] COMPILING TO EXECUTION GRAPH...")
    graph = compile_task(task)
    print(f"  Nodes: {len(graph.nodes)}")
    levels = graph.get_execution_order()
    for i, level in enumerate(levels):
        names = [n.result_key or n.func for n in level]
        print(f"  Level {i}: {names}")

    print("\n[4] EXECUTING...")
    result = run(graph, inputs={"camera": "camera_01", "target_id": "vessel_42"})

    print("\n[5] EXECUTION LOG:")
    print("-" * 40)
    for entry in result.log:
        print(f"  {entry}")

    print("\n[6] RESULTS:")
    print("-" * 40)
    print(f"  Success: {result.success}")
    print(f"  LLM calls: {result.total_llm_calls}")
    print(f"  LLM tokens: {result.total_llm_tokens}")
    print(f"  Constraint violations: {result.constraint_violations}")
    print(f"  Errors: {result.errors}")
    print(f"\n  Outputs:")
    for k, v in result.outputs.items():
        print(f"    {k}: {v}")

    print("\n[7] INTERNAL STATE:")
    print("-" * 40)
    for k, v in result.state.items():
        print(f"  {k}: {json.dumps(v, default=str)[:100]}")

    print("\n" + "=" * 60)
    print(f"Task '{result.task_name}' completed: {'PASS' if result.success else 'FAIL'}")
    print("=" * 60)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
