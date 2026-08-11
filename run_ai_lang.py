"""Run robot navigation through the AI Lang interpreter.

The AI Lang program is the ONLY code. No Python orchestration.
"""
import sys
sys.path.insert(0, "src")

from ai_lang import execute
from ai_lang.skills import sim2d, rl_dqn


def main():
    print("=" * 60)
    print("AI Lang — Robot Navigation (Full Interpreter)")
    print("=" * 60)

    # Load the AI Lang program
    with open("examples/robot_navigation.ais", "r") as f:
        source = f.read()

    print("\n--- PROGRAM ---")
    for i, line in enumerate(source.strip().splitlines(), 1):
        print(f"  {i:3}: {line}")
    print("--- END ---\n")

    # Register skills
    extra_skills = {
        "create_world": sim2d.create_world,
        "train_episode": rl_dqn.train_episode,
        "evaluate_agent": rl_dqn.evaluate_agent,
        "create_agent": rl_dqn.create_agent,
        "flatten_obs": rl_dqn._flatten_obs,
        "sum": lambda lst: sum(lst) if lst else 0,
        "length": len,
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
    }

    # Execute entirely through AI Lang
    print("[EXECUTING]")
    result = execute(source, inputs=None, extra_skills=extra_skills)

    print("\n--- RESULT ---")
    print(f"Success: {result['success']}")
    print(f"Task: {result['task_name']}")

    print(f"\nExecution Log:")
    for entry in result.get("log", []):
        print(f"  {entry}")

    print(f"\nOutputs:")
    for name, value in result.get("outputs", {}).items():
        print(f"  {name}: {value}")

    print(f"\nKey Variables:")
    for name in ["success_rate", "episodes", "agent", "world"]:
        if name in result.get("variables", {}):
            val = result["variables"][name]
            print(f"  {name}: {val}")

    print("\n" + "=" * 60)
    if result["success"]:
        print("RESULT: PASS — AI Lang program executed successfully!")
    else:
        print("RESULT: FAIL — program did not pass verification")
    print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
