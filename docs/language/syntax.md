# Language Syntax Reference

## Program Structure

A AI Lang program consists of a single `TASK` block:

```ais
TASK task_name {
    INPUT { ... }
    COMPUTE { ... }
    OUTPUT { ... }
    VERIFY { ... }
}
```

## Blocks

| Block | Purpose |
|-------|---------|
| `INPUT` | Declare input parameters with optional defaults |
| `COMPUTE` | Executable computation logic |
| `OUTPUT` | Declare output variables |
| `VERIFY` | Check constraints (fails task if false) |
| `PERCEIVE` | Sensor/data ingestion operations |
| `REASON` | Cognitive/reasoning operations |
| `PLAN` | Planning with objectives and constraints |
| `CONTROL` | Action/motor command generation |
| `EVALUATE` | Post-computation evaluation |

## INPUT Block

```ais
INPUT {
    episodes = 100,
    learning_rate = 0.001,
    model = "llama-3.3-70b-versatile",
    seed = 42
}
```

Supports typed declarations and default values.

## COMPUTE Block

The `COMPUTE` block contains executable statements:

```ais
COMPUTE {
    # Assignment
    x = 10
    y = x * 2 + 5

    # Function calls
    world = create_world(42)
    agent = create_agent(38, 9)

    # FOR loops
    FOR ep IN 1..episodes {
        result = train_episode(agent, world)
    }

    # IF/ELSE
    IF result.success {
        successes = successes + 1
    } ELSE {
        failures = failures + 1
    }

    # Logging with interpolation
    LOG "Episode {ep}: reward={result.reward}"
}
```

## Expressions

### Arithmetic
```
a + b, a - b, a * b, a / b, a % b
```

### Comparison
```
a < b, a > b, a <= b, a >= b, a == b, a != b
```

### Logical
```
a AND b, a OR b, NOT a
```

### Field Access
```
result.success, agent.epsilon, eval_results.success_rate
```

### Array/List Access
```
rewards[0], rewards[-1], rewards[1..50]
```

## FOR Loops

```ais
FOR variable IN start..end {
    # body
}
```

- `start` and `end` are inclusive
- Supports expressions: `FOR ep IN 1..episodes`
- Nested loops supported

## IF/ELSE

```ais
IF condition {
    # true branch
} ELSE {
    # false branch
}
```

## OUTPUT Block

```ais
OUTPUT {
    success_rate,
    avg_reward,
    model_path
}
```

Lists variable names to expose as outputs.

## VERIFY Block

```ais
VERIFY {
    CHECK success_rate > 0.2
    CHECK avg_reward > -10
}
```

If any CHECK fails, the task is marked as failed.
