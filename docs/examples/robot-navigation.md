# Robot Navigation Example

Full example of training a 2D robot to navigate using Deep Q-Network.

## Program

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

        eval_results = evaluate_agent(agent, world, eval_episodes)
        success_rate = eval_results.success_rate
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
print(f"Success rate: {result['outputs']['success_rate']}")
```

## See Also

- [Robot Navigation Tutorial](../tutorials/robot-navigation.md)
