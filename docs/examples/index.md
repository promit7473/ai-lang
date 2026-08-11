# Examples

## [Robot Navigation](robot-navigation.md)

Train a 2D mobile robot using Deep Q-Network reinforcement learning.

```ais
TASK robot_navigation {
    INPUT { episodes = 400, seed = 42 }
    COMPUTE {
        world = create_world(seed)
        agent = create_agent(38, 9)
        FOR ep IN 1..episodes {
            train_episode(agent, world, 150)
        }
    }
}
```

## [LLM Orchestration](llm-orchestration.md)

Chain multiple LLM calls with verification.

```ais
TASK analyze {
    PERCEIVE { objects = detect_objects(image) }
    REASON { report = generate_report(objects) }
    VERIFY { CHECK confidence > 0.8 }
}
```

## [Multi-Agent Pipeline](multi-agent.md)

Coordinate perception, planning, and control agents.

```ais
TASK autonomous_driving {
    PERCEIVE {
        camera = camera_capture()
        lidar = lidar_scan()
    }
    REASON {
        objects = detect_objects(camera)
        path = plan_path(objects, lidar)
    }
    CONTROL {
        command = generate_control(path)
    }
}
```
