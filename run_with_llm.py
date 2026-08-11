"""Run the vessel_monitor example with a real LLM via OpenRouter.

Usage:
    $env:OPENROUTER_API_KEY="your-key"; python run_with_llm.py

Optional env vars:
    OPENROUTER_MODEL   — model to use (default: google/gemini-2.0-flash-001)
"""
import sys
import os
sys.path.insert(0, "src")

from ai_lang import from_env, parse, compile_task, run


def main():
    llm = from_env()

    if llm.__class__.__name__ == "MockLLM":
        print("WARNING: No API key found. Set OPENROUTER_API_KEY env var.")
        print("Falling back to mock LLM.\n")

    print(f"Using LLM: {llm.__class__.__name__}")
    if hasattr(llm, 'model'):
        print(f"Model: {llm.model}")

    with open("examples/vessel_monitor.ais", "r") as f:
        source = f.read()

    task = parse(source)
    graph = compile_task(task)

    print(f"\nTask: {task.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Objectives: {[(o.type.value, o.metric) for o in graph.objectives]}")
    print(f"Constraints: {[(c.metric, c.op.value, c.value) for c in graph.constraints]}")
    print()

    result = run(graph, inputs={"camera": "camera_01", "target_id": "vessel_42"}, llm=llm, use_llm=True)

    print("\n--- EXECUTION LOG ---")
    for entry in result.log:
        print(f"  {entry}")

    print(f"\n--- RESULT ---")
    print(f"Success: {result.success}")
    print(f"LLM calls: {result.total_llm_calls}")
    print(f"Total tokens: {result.total_llm_tokens}")
    print(f"Constraint violations: {result.constraint_violations}")
    print(f"\nOutputs:")
    for k, v in result.outputs.items():
        print(f"  {k}: {v}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
