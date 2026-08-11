"""End-to-end robot navigation training with AI Lang.

This runner interprets the robot_navigation.ais program
and executes it using the sim2d and rl_dqn skills.
"""
import sys
import time
sys.path.insert(0, "src")

from ai_lang.skills.sim2d import create_world, render_world, evaluate_policy
from ai_lang.skills.rl_dqn import (
    create_agent, train_episode, evaluate_agent, _flatten_obs
)


def main():
    print("=" * 60)
    print("AI Lang — Robot Navigation RL Training")
    print("=" * 60)

    config = {
        "episodes": 800,
        "eval_episodes": 10,
        "seed": 42,
        "state_size": None,
        "action_size": 9,
        "max_steps": 200,
        "render_every": 100,
        "render_path": "renders",
    }

    print(f"\nConfig: { {k:v for k,v in config.items() if k != 'render_path'} }")

    import os
    os.makedirs(config["render_path"], exist_ok=True)

    print("\n[1] Creating world...")
    world = create_world(seed=config["seed"])
    print(f"  World: {world.config.width}x{world.config.height}")
    print(f"  Obstacles: {len(world.obstacles)}")
    print(f"  Goal: ({world.goal[0]:.1f}, {world.goal[1]:.1f})")

    render_path = os.path.join(config["render_path"], "initial.png")
    render_world(world, save_path=render_path)
    print(f"  Initial state rendered: {render_path}")

    print("\n[2] Computing state size...")
    test_obs = world.reset()
    test_state = _flatten_obs(test_obs)
    config["state_size"] = len(test_state)
    print(f"  Computed state size: {config['state_size']}")

    print("\n[3] Creating DQN agent...")
    agent = create_agent(
        state_size=config["state_size"],
        action_size=config["action_size"],
        hidden_sizes=[64, 64],
        learning_rate=0.002,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        batch_size=128,
        target_update_freq=20,
        buffer_capacity=20000,
    )
    print(f"  State size: {config['state_size']}")
    print(f"  Action size: {config['action_size']}")
    print(f"  Network: {config['state_size']} -> [64, 64] -> {config['action_size']}")

    print(f"\n[4] Training for {config['episodes']} episodes...")
    start_time = time.time()
    rewards_history = []
    success_history = []

    for episode in range(1, config["episodes"] + 1):
        result = train_episode(agent, world, max_steps=config["max_steps"])
        rewards_history.append(result["reward"])
        success_history.append(1 if result["success"] else 0)

        if episode % 50 == 0 or episode == 1:
            recent_success = sum(success_history[-50:]) / min(50, len(success_history))
            recent_reward = sum(rewards_history[-50:]) / min(50, len(rewards_history))
            stats = agent.get_stats()
            print(
                f"  Episode {episode:4d} | "
                f"Reward: {result['reward']:7.1f} | "
                f"Avg50: {recent_reward:7.1f} | "
                f"Success50: {recent_success:.0%} | "
                f"Eps: {stats['epsilon']:.3f} | "
                f"Steps: {result['steps']}"
            )

        if episode % config["render_every"] == 0:
            render_world(world, save_path=os.path.join(config["render_path"], f"episode_{episode}.png"))

    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.1f}s")

    print("\n[5] Evaluating trained policy...")
    eval_results = evaluate_agent(agent, world, num_episodes=config["eval_episodes"], max_steps=config["max_steps"])
    print(f"  Success rate: {eval_results['success_rate']:.0%}")
    print(f"  Avg reward: {eval_results['avg_reward']:.1f}")
    print(f"  Successes: {eval_results['successes']}/{config['eval_episodes']}")
    print(f"  Collisions: {eval_results['collisions']}/{config['eval_episodes']}")
    print(f"  Timeouts: {eval_results['timeouts']}/{config['eval_episodes']}")

    print("\n[6] Rendering final trajectory...")
    obs = world.reset()
    done = False
    steps = 0
    trajectory_reward = 0.0

    while not done and steps < config["max_steps"]:
        state = _flatten_obs(obs)
        if len(state) < config["state_size"]:
            state.extend([0.0] * (config["state_size"] - len(state)))
        action = agent.select_action(state, evaluate=True)
        obs, reward, done, info = world.step(action)
        trajectory_reward += reward
        steps += 1

    final_render = os.path.join(config["render_path"], "final_trajectory.png")
    render_world(world, save_path=final_render)
    print(f"  Final trajectory: {steps} steps, reward={trajectory_reward:.1f}")
    print(f"  Rendered: {final_render}")

    print("\n[7] Training summary:")
    print(f"  Total episodes: {config['episodes']}")
    print(f"  Total time: {train_time:.1f}s")
    print(f"  Episodes/sec: {config['episodes']/train_time:.1f}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    print(f"  Total training steps: {agent.steps}")

    print("\n" + "=" * 60)
    if eval_results["success_rate"] > 0.3:
        print("RESULT: PASS — Robot learned to navigate!")
    else:
        print("RESULT: NEEDS MORE TRAINING")
    print("=" * 60)

    return 0 if eval_results["success_rate"] > 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
