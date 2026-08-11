"""Deep Q-Network (DQN) Reinforcement Learning Skill.

CPU-friendly RL for robot navigation.
Implements DQN with experience replay and target network.
"""
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Transition:
    state: list
    action: int
    reward: float
    next_state: list
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int] = None):
        import torch
        import torch.nn as nn
        hidden_sizes = hidden_sizes or [64, 64]
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        self._torch = torch

    def forward(self, x):
        if not isinstance(x, self._torch.Tensor):
            x = self._torch.tensor(x, dtype=self._torch.float32)
        return self.model(x)

    def predict(self, state: list) -> list:
        with self._torch.no_grad():
            q_values = self.forward(state)
            return q_values.numpy().tolist() if hasattr(q_values, 'numpy') else q_values.tolist()

    def get_action(self, state: list, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.model[-1].out_features - 1)
        q_values = self.predict(state)
        return q_values.index(max(q_values))

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def __call__(self, x):
        return self.forward(x)


@dataclass
class DQNAgent:
    state_size: int
    action_size: int
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 0.005
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    batch_size: int = 128
    target_update_freq: int = 5
    buffer_capacity: int = 20000

    def __post_init__(self):
        import torch.optim as optim
        self.q_network = QNetwork(self.state_size, self.action_size, self.hidden_sizes)
        self.target_network = QNetwork(self.state_size, self.action_size, self.hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.epsilon = self.epsilon_start
        self.steps = 0
        self.episodes = 0
        self.losses = []

    def select_action(self, state: list, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.q_network.get_action(state, epsilon=0.0)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        import torch
        import torch.nn as nn

        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> dict:
        recent_loss = sum(self.losses[-100:]) / max(len(self.losses[-100:]), 1)
        return {
            "episodes": self.episodes,
            "steps": self.steps,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.replay_buffer),
            "avg_loss": round(recent_loss, 6),
        }


def create_agent(state_size: int, action_size: int, **kwargs) -> DQNAgent:
    return DQNAgent(state_size=state_size, action_size=action_size, **kwargs)


def train_episode(agent: DQNAgent, world, max_steps: int = 500,
                  obs_to_state: Callable = None) -> dict:
    obs = world.reset()
    episode_reward = 0.0
    steps = 0
    done = False

    if obs_to_state is None:
        obs_to_state = lambda o: _flatten_obs(o)

    while not done and steps < max_steps:
        state = obs_to_state(obs)
        action = agent.select_action(state)
        next_obs, reward, done, info = world.step(action)
        next_state = obs_to_state(next_obs)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        episode_reward += reward
        obs = next_obs
        steps += 1

    agent.end_episode()
    return {
        "reward": episode_reward,
        "steps": steps,
        "success": info.get("reached_goal", False),
        "collision": info.get("collided", False),
    }


def _flatten_obs(obs: dict) -> list:
    state = []
    state.extend(obs.get("lidar", []))
    state.extend(obs.get("lidar_hits", []))
    state.append(obs.get("distance_to_goal", 0.0) / 10.0)
    state.append(obs.get("angle_to_goal", 0.0) / math.pi)
    state.append(obs.get("linear_vel", 0.0) / 2.0)
    state.append(obs.get("angular_vel", 0.0) / 1.5)
    state.append(math.sin(obs.get("heading", 0.0)))
    state.append(math.cos(obs.get("heading", 0.0)))
    return state


def evaluate_agent(agent: DQNAgent, world, num_episodes: int = 10,
                   max_steps: int = 500) -> dict:
    from .sim2d import evaluate_policy

    state_size = agent.state_size
    action_size = agent.action_size

    def policy_fn(obs):
        state = _flatten_obs(obs)
        if len(state) < state_size:
            state.extend([0.0] * (state_size - len(state)))
        elif len(state) > state_size:
            state = state[:state_size]
        return agent.select_action(state, evaluate=True)

    return evaluate_policy(world, policy_fn, num_episodes)
