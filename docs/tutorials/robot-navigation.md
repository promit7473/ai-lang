# Robot Navigation Tutorial

This tutorial trains a 2D mobile robot to navigate to a goal while avoiding obstacles using Deep Q-Network (DQN) reinforcement learning — all written in AI Lang.

## The AI Lang Program

```ais
TASK robot_navigation {
    INPUT {
        episodes = 400,
        max_steps = 150,
        eval_episodes = 5,
        seed = 42
    }

    COMPUTE {
        world = create_world(seed)
        state_size = 38
        action_size = 9
        agent = create_agent(state_size, action_size)

        successes = 0

        FOR ep IN 1..episodes {
            result = train_episode(agent, world, max_steps)

            IF result.success {
                successes = successes + 1
            }

            IF ep % 50 == 0 {
                ep_float = ep
                success_rate = successes / ep_float
                LOG "Episode {ep}: successes={successes}, rate={success_rate}"
            }
        }

        LOG "Training complete. Evaluating..."
        eval_results = evaluate_agent(agent, world, eval_episodes)
        success_rate = eval_results.success_rate
        LOG "Final success rate: {success_rate}"
    }

    OUTPUT {
        success_rate,
        eval_results
    }

    VERIFY {
        CHECK success_rate >= 0.0
    }
}
```

## How It Works

### 1. World Creation
`create_world(seed)` generates a 2D environment with:
- Random obstacles
- A goal position
- A robot with LiDAR sensors

### 2. Agent Creation
`create_agent(state_size, action_size)` creates a DQN agent with:
- Experience replay buffer
- Target network
- Epsilon-greedy exploration

### 3. Training Loop
```
FOR ep IN 1..episodes {
    result = train_episode(agent, world, max_steps)
    ...
}
```

Each episode:
1. Reset world
2. Get observation (LiDAR distances, goal direction)
3. Select action (epsilon-greedy)
4. Step physics, get reward
5. Store transition in replay buffer
6. Train Q-network on mini-batch

### 4. Reward Function
```python
reward = (prev_dist - curr_dist) * 5.0  # Progress toward goal
if min_lidar < 1.0:
    reward -= 2.0  # Too close to obstacle
if reached_goal:
    reward += 100.0
if collided:
    reward -= 20.0
```

### 5. State Representation
```
[lidar_1, ..., lidar_16, hits_1, ..., hits_16, distance_to_goal, angle_to_goal, linear_vel, angular_vel, sin(heading), cos(heading)]
```
Total: 38 dimensions

### 6. Action Space
| Action | Description |
|--------|-------------|
| 0 | Full speed forward |
| 1 | Half speed + turn right |
| 2 | Half speed + turn left |
| 3 | Rotate right |
| 4 | Rotate left |
| 5 | Slow backward |
| 70 | Fast forward + slight right |
| 7 | Fast forward + slight left |
| 8 | Stop |

## Running

```python
from ai_lang import execute
from ai_lang.skills import sim2d, rl_dqn

skills = {
    "create_world": sim2d.create_world,
    "train_episode": rl_dqn.train_episode,
    "evaluate_agent": rl_dqn.evaluate_agent,
    "create_agent": rl_dqn.create_agent,
}

result = execute(open("robot_navigation.ais").read(), extra_skills=skills)
print(result["outputs"]["success_rate"])
```

## Expected Output

```
Episode 50: successes=0, rate=0.00
Episode 100: successes=0, rate=0.00
Episode 150: successes=1, rate=0.01
Episode 200: successes=2, rate=0.01
Episode 250: successes=5, rate=0.02
Episode 300: successes=12, rate=0.04
Episode 350: successes=28, rate=0.08
Episode 400: successes=52, rate=0.13
Training complete. Evaluating...
Final success rate: 0.20
```

## Next Steps

- Add ROS2 skills for physical robot deployment
- Extend to 3D with Isaac Sim
- Multi-agent coordination
