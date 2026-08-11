# Blocks Reference

## COMPUTE Block

The main execution block. Contains statements that run sequentially.

```ais
COMPUTE {
    x = 10
    y = x * 2
    FOR i IN 1..5 {
        y = y + 1
    }
}
```

## PERCEIVE Block

Sensor and data ingestion operations.

```ais
PERCEIVE {
    objects = detect_objects(camera)
    tracks = track_objects(objects)
}
```

## REASON Block

Cognitive and reasoning operations.

```ais
REASON {
    state = estimate_state(tracks)
    trajectory = predict_path(state, horizon=5.0)
    risk = assess_collision(trajectory)
}
```

## PLAN Block

Planning with objectives and constraints.

```ais
PLAN {
    OBJECTIVE minimize risk
    OBJECTIVE maximize tracking_accuracy
    CONSTRAINT risk < 0.3
}
```

## CONTROL Block

Action and motor command generation.

```ais
CONTROL {
    command = generate_command(state, trajectory)
}
```

## EVALUATE Block

Post-computation evaluation.

```ais
EVALUATE {
    results = evaluate_agent(agent, world, 10)
    success_rate = results.success_rate
}
```

## VERIFY Block

Constraint checking. Fails the task if any check is false.

```ais
VERIFY {
    CHECK success_rate > 0.2
    CHECK avg_reward > -10
}
```
