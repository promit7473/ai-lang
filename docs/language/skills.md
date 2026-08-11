# Skills

Skills are reusable capabilities that AI Lang programs can call.

## Built-in Skills

### Robotics Skills (`ai_lang.skills.robotics`)

| Skill | Description |
|-------|-------------|
| `detect_objects(source)` | Detect objects in a sensor feed |
| `track_objects(objects, source)` | Track objects across frames |
| `estimate_state(track)` | Estimate kinematic state |
| `predict_path(state, horizon)` | Predict future trajectory |
| `assess_collision(trajectory)` | Assess collision risk |
| `generate_command(state, trajectory)` | Generate control command |
| `localize(landmarks)` | Localize robot position |

### Simulation Skills (`ai_lang.skills.sim2d`)

| Skill | Description |
|-------|-------------|
| `create_world(seed)` | Create a 2D simulation world |
| `step_world(world, action)` | Step the simulation |
| `reset_world(world)` | Reset to initial state |
| `render_world(world, path)` | Render to image |
| `evaluate_policy(world, policy, n)` | Evaluate a policy |

### RL Skills (`ai_lang.skills.rl_dqn`)

| Skill | Description |
|-------|-------------|
| `create_agent(state_size, action_size, **kwargs)` | Create DQN agent |
| `train_episode(agent, world, max_steps)` | Train for one episode |
| `evaluate_agent(agent, world, n)` | Evaluate trained agent |

## Custom Skills

Register custom skills:

```python
from ai_lang import execute
from ai_lang.skills import register_skill

register_skill(
    name="my_skill",
    func="my_func",
    category="custom",
    description="Does something useful",
    implementation=lambda x: x * 2,
)

result = execute(program, extra_skills={"my_skill": my_impl})
```
