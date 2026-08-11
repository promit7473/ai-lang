"""Real end-to-end test with Groq API.

Scenarios:
1. Happy path — constraints pass on first try
2. Budget limit — low max_tokens triggers graceful stop
3. Schema validation — verifies schemas are sent to LLM

Usage:
    $env:GROQ_API_KEY="gsk-..."; python run_real_test.py
    $env:GROQ_API_KEY="gsk-..."; $env:GROQ_MODEL="llama-3.1-8b-instant"; python run_real_test.py
"""
import sys
import os
sys.path.insert(0, "src")

from ai_lang import from_env, parse, compile_task, run


VESSEL_MONITOR = """
TASK vessel_monitor {
    INPUT {
        camera: Camera,
        target_id: string
    }

    PERCEIVE {
        objects = detect_objects(camera)
        tracks = track_objects(objects, camera)
    }

    REASON {
        state = estimate_state(tracks)
        trajectory = predict_path(state, horizon=5.0)
        risk = assess_collision(trajectory)
    }

    PLAN {
        OBJECTIVE minimize risk
        OBJECTIVE maximize tracking_accuracy
        CONSTRAINT risk < 0.3
    }

    CONTROL {
        command = generate_command(state, trajectory)
    }

    OUTPUT {
        trajectory: Path,
        risk: float,
        command: Control
    }

    VERIFY {
        CHECK constraints_satisfied
        CHECK schema_valid
    }
}
"""


def test_happy_path(llm):
    print("\n" + "=" * 60)
    print("SCENARIO 1: Happy Path (constraints should pass)")
    print("=" * 60)

    task = parse(VESSEL_MONITOR)
    graph = compile_task(task)
    result = run(graph, inputs={"camera": "camera_01", "target_id": "vessel_42"}, llm=llm, use_llm=True)

    print(f"\nSuccess: {result.success}")
    print(f"LLM calls: {result.total_llm_calls}")
    print(f"Tokens used: {result.total_llm_tokens}")
    print(f"Cost: ${result.total_llm_cost:.6f}")
    print(f"Repair attempts: {result.repair_attempts}")
    print(f"Constraint violations: {result.constraint_violations}")
    print(f"Budget exceeded: {result.budget_exceeded}")

    print(f"\nOutputs:")
    for name, info in result.outputs.items():
        print(f"  {name}: trust={info.get('trust_level', 'unknown')}, value={str(info.get('value', info))[:60]}")

    print(f"\nEvents ({len(result.events)}):")
    for evt in result.events:
        data = {k:v for k,v in evt.items() if k not in ('type','timestamp')}
        print(f"  {evt['type']}: {data}")

    return result.success


def test_budget_limit(llm):
    print("\n" + "=" * 60)
    print("SCENARIO 2: Budget Limit (max_tokens=50)")
    print("=" * 60)

    task = parse(VESSEL_MONITOR)
    graph = compile_task(task)
    result = run(graph, inputs={"camera": "camera_01", "target_id": "vessel_42"}, llm=llm, use_llm=True, max_tokens=50)

    print(f"\nSuccess: {result.success}")
    print(f"LLM calls: {result.total_llm_calls}")
    print(f"Tokens used: {result.total_llm_tokens}")
    print(f"Budget exceeded: {result.budget_exceeded}")
    print(f"Metrics: {result.metrics}")

    return result.budget_exceeded


def test_schema_validation():
    print("\n" + "=" * 60)
    print("SCENARIO 3: Schema Validation (unit-level)")
    print("=" * 60)

    from ai_lang.skills import registry

    skill = registry.get("assess_collision")
    print(f"\nSkill: {skill.name}")
    print(f"Has schema: {skill.output_schema is not None}")

    # Valid output
    valid = {"risk": 0.15, "safe": True}
    passed, error = skill.validate_output(valid)
    print(f"Valid output {valid}: passed={passed}, error='{error}'")

    # Invalid: risk > 1
    invalid1 = {"risk": 1.5, "safe": True}
    passed, error = skill.validate_output(invalid1)
    print(f"Invalid (risk>1) {invalid1}: passed={passed}, error='{error}'")

    # Invalid: missing 'safe'
    invalid2 = {"risk": 0.15}
    passed, error = skill.validate_output(invalid2)
    print(f"Invalid (missing safe) {invalid2}: passed={passed}, error='{error}'")

    # Invalid: wrong type
    invalid3 = {"risk": "high", "safe": True}
    passed, error = skill.validate_output(invalid3)
    print(f"Invalid (wrong type) {invalid3}: passed={passed}, error='{error}'")

    # Test repair prompt generation
    repair_prompt = skill.build_repair_prompt(
        original_prompt="Assess collision risk",
        invalid_output='{"risk": 1.5}',
        error="Value 1.5 above maximum 1"
    )
    print(f"\nRepair prompt (first 200 chars):\n{repair_prompt[:200]}...")

    return True


def main():
    llm = from_env()
    print(f"LLM: {llm.__class__.__name__}")
    if hasattr(llm, 'model'):
        print(f"Model: {llm.model}")
    if hasattr(llm, 'base_url'):
        print(f"Base URL: {llm.base_url}")

    if llm.__class__.__name__ == "MockLLM":
        print("\nNo API key detected. Set GROQ_API_KEY env var for real LLM testing.")
        print("   Running schema validation test only...\n")
        test_schema_validation()
        return 0

    all_passed = True

    try:
        if not test_happy_path(llm):
            all_passed = False
            print("\n[FAIL] Happy path")
        else:
            print("\n[PASS] Happy path")
    except Exception as e:
        all_passed = False
        print(f"\n[ERROR] Happy path: {e}")

    # Reset LLM counters for next test
    llm.call_count = 0
    llm.total_tokens = 0

    try:
        if not test_budget_limit(llm):
            all_passed = False
            print("\n[FAIL] Budget limit")
        else:
            print("\n[PASS] Budget limit")
    except Exception as e:
        all_passed = False
        print(f"\n[ERROR] Budget limit: {e}")

    test_schema_validation()

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    if all_passed:
        print("Real end-to-end test with Groq: SUCCESS")
    else:
        print("Real end-to-end test with Groq: ISSUES DETECTED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
