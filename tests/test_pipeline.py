import sys
sys.path.insert(0, "src")

from ai_lang import parse, compile_task, run
from ai_lang.skills import registry
from ai_lang.ir import IROp, ConstraintOp, ObjectiveType


def test_lexer_parser_basic():
    source = """
    TASK test_task {
        INPUT {
            sensor: Camera
        }
        PERCEIVE {
            result = detect_objects(sensor)
        }
        OUTPUT {
            result: Objects
        }
    }
    """
    task = parse(source)
    assert task.name == "test_task"
    assert len(task.inputs) == 1
    assert task.inputs[0].name == "sensor"
    assert task.inputs[0].type_name == "Camera"
    assert len(task.blocks) == 1
    assert task.blocks[0].op == IROp.PERCEIVE


def test_parser_objectives_constraints():
    source = """
    TASK planner {
        PLAN {
            OBJECTIVE minimize cost
            OBJECTIVE maximize speed
            CONSTRAINT cost < 100
            CONSTRAINT latency <= 50ms
        }
        OUTPUT {
            plan: Path
        }
    }
    """
    task = parse(source)
    plan_block = task.get_block(IROp.PLAN)
    assert plan_block is not None
    assert len(plan_block.objectives) == 2
    assert plan_block.objectives[0].type == ObjectiveType.MINIMIZE
    assert plan_block.objectives[0].metric == "cost"
    assert plan_block.objectives[1].type == ObjectiveType.MAXIMIZE
    assert len(plan_block.constraints) == 2
    assert plan_block.constraints[0].op == ConstraintOp.LT
    assert plan_block.constraints[0].value == 100
    assert plan_block.constraints[1].op == ConstraintOp.LE
    assert plan_block.constraints[1].value == 50
    assert plan_block.constraints[1].unit == "ms"


def test_compiler_dependency_graph():
    source = """
    TASK dep_test {
        INPUT {
            cam: Camera
        }
        PERCEIVE {
            a = detect_objects(cam)
            b = track_objects(a, cam)
        }
        REASON {
            c = estimate_state(b)
        }
        OUTPUT {
            c: State
        }
    }
    """
    task = parse(source)
    graph = compile_task(task)
    levels = graph.get_execution_order()

    assert len(levels) == 3
    level_names = [[n.result_key for n in level] for level in levels]
    assert level_names[0] == ["a"]
    assert level_names[1] == ["b"]
    assert level_names[2] == ["c"]


def test_runtime_execution():
    source = """
    TASK exec_test {
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
        }
        OUTPUT {
            state: State
        }
    }
    """
    task = parse(source)
    graph = compile_task(task)
    result = run(graph, inputs={"camera": "cam_01", "target_id": "t1"})

    assert result.success
    assert "state" in result.outputs
    assert "state" in result.state


def test_runtime_constraint_violation():
    source = """
    TASK constraint_test {
        PERCEIVE {
            objects = detect_objects("camera")
            tracks = track_objects(objects, "camera")
        }
        REASON {
            state = estimate_state(tracks)
            traj = predict_path(state, horizon=5.0)
            risk = assess_collision(traj)
        }
        PLAN {
            CONSTRAINT risk < 0.1
        }
        OUTPUT {
            risk: float
        }
    }
    """
    task = parse(source)
    graph = compile_task(task)
    result = run(graph, inputs={})

    assert not result.success
    assert len(result.constraint_violations) > 0


def test_skill_registry():
    assert registry.has("detect_objects")
    assert registry.has("track_objects")
    assert registry.has("estimate_state")
    assert registry.has("predict_path")
    assert registry.has("assess_collision")
    assert registry.has("generate_command")
    assert registry.has("localize")
    assert not registry.has("nonexistent_skill")


def test_verify_block():
    source = """
    TASK verify_test {
        PERCEIVE {
            objects = detect_objects("cam")
        }
        VERIFY {
            CHECK constraints_satisfied
            CHECK schema_valid
        }
        OUTPUT {
            objects: Objects
        }
    }
    """
    task = parse(source)
    verify_block = task.get_block(IROp.VERIFY)
    assert verify_block is not None
    assert len(verify_block.checks) == 2
    assert verify_block.checks[0].name == "constraints_satisfied"
    assert verify_block.checks[1].name == "schema_valid"


def test_type_names_not_keyword():
    source = """
    TASK types_test {
        INPUT {
            camera: Camera
        }
        PERCEIVE {
            x = detect_objects(camera)
        }
        OUTPUT {
            x: Control
        }
    }
    """
    task = parse(source)
    assert task.outputs[0].type_name == "Control"


if __name__ == "__main__":
    tests = [
        test_lexer_parser_basic,
        test_parser_objectives_constraints,
        test_compiler_dependency_graph,
        test_runtime_execution,
        test_runtime_constraint_violation,
        test_skill_registry,
        test_verify_block,
        test_type_names_not_keyword,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
