"""2D Robot Navigation Simulator Skill.

Lightweight CPU-only 2D simulator for robot navigation with:
- Differential drive physics
- LiDAR raycasting (distance sensors)
- Obstacle collision detection
- Matplotlib rendering
- Gymnasium-compatible interface for RL
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    linear_vel: float = 0.0
    angular_vel: float = 0.0


@dataclass
class LidarReading:
    angle: float
    distance: float
    hit: bool


@dataclass
class SimConfig:
    width: float = 12.0
    height: float = 12.0
    robot_radius: float = 0.3
    max_speed: float = 2.0
    max_angular_speed: float = 1.5
    dt: float = 0.1
    num_lidar_rays: int = 16
    lidar_range: float = 5.0
    lidar_fov: float = 2 * math.pi
    goal_radius: float = 1.0
    max_steps: int = 300


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float


class World2D:
    def __init__(self, config: SimConfig = None, seed: int = None):
        self.config = config or SimConfig()
        self.rng = random.Random(seed)
        self.robot = RobotState(x=2.0, y=2.0, heading=0.0)
        self.obstacles: list[Obstacle] = []
        self.goal = (self.config.width - 2.0, self.config.height - 2.0)
        self.steps = 0
        self.max_steps = 500
        self.collided = False
        self.reached_goal = False
        self._generate_obstacles()

    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = self.rng.randint(5, 12)
        for _ in range(num_obstacles):
            w = self.rng.uniform(0.5, 2.5)
            h = self.rng.uniform(0.5, 2.5)
            x = self.rng.uniform(3.0, self.config.width - 3.0)
            y = self.rng.uniform(3.0, self.config.height - 3.0)
            if abs(x - self.robot.x) < 3.0 and abs(y - self.robot.y) < 3.0:
                continue
            if abs(x - self.goal[0]) < 2.0 and abs(y - self.goal[1]) < 2.0:
                continue
            self.obstacles.append(Obstacle(x=x, y=y, width=w, height=h))

    def reset(self) -> dict:
        self.robot = RobotState(
            x=self.rng.uniform(1.0, 4.0),
            y=self.rng.uniform(1.0, 4.0),
            heading=self.rng.uniform(0, 2 * math.pi)
        )
        self.goal = (
            self.rng.uniform(self.config.width - 4.0, self.config.width - 1.0),
            self.rng.uniform(self.config.height - 4.0, self.config.height - 1.0)
        )
        self._generate_obstacles()
        self.steps = 0
        self.collided = False
        self.reached_goal = False
        return self.get_observation()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        linear, angular = self._action_to_vel(action)
        self.robot.linear_vel = linear
        self.robot.angular_vel = angular

        new_x = self.robot.x + linear * math.cos(self.robot.heading) * self.config.dt
        new_y = self.robot.y + linear * math.sin(self.robot.heading) * self.config.dt
        new_heading = self.robot.heading + angular * self.config.dt
        new_heading = new_heading % (2 * math.pi)

        self.robot.x = new_x
        self.robot.y = new_y
        self.robot.heading = new_heading

        self._check_collision()
        self._check_bounds()
        self.steps += 1

        reward = self._compute_reward(action)
        done = self.collided or self.reached_goal or self.steps >= self.max_steps
        info = {
            "distance_to_goal": self._distance_to_goal(),
            "collided": self.collided,
            "reached_goal": self.reached_goal,
            "steps": self.steps,
        }

        return self.get_observation(), reward, done, info

    def get_observation(self) -> dict:
        lidar = self.get_lidar()
        dist_to_goal = self._distance_to_goal()
        angle_to_goal = self._angle_to_goal()
        return {
            "lidar": [reading.distance for reading in lidar],
            "lidar_hits": [1.0 if reading.hit else 0.0 for reading in lidar],
            "robot_x": self.robot.x,
            "robot_y": self.robot.y,
            "heading": self.robot.heading,
            "linear_vel": self.robot.linear_vel,
            "angular_vel": self.robot.angular_vel,
            "goal_x": self.goal[0],
            "goal_y": self.goal[1],
            "distance_to_goal": dist_to_goal,
            "angle_to_goal": angle_to_goal,
        }

    def get_lidar(self) -> list[LidarReading]:
        readings = []
        for i in range(self.config.num_lidar_rays):
            angle = self.robot.heading + (i / self.config.num_lidar_rays) * self.config.lidar_fov
            dist = self._raycast(angle)
            hit = dist < self.config.lidar_range
            readings.append(LidarReading(angle=angle, distance=dist, hit=hit))
        return readings

    def _raycast(self, angle: float) -> float:
        min_dist = self.config.lidar_range
        step_size = 0.1
        for d in range(int(self.config.lidar_range / step_size)):
            dist = d * step_size
            px = self.robot.x + dist * math.cos(angle)
            py = self.robot.y + dist * math.sin(angle)
            if px < 0 or px > self.config.width or py < 0 or py > self.config.height:
                return dist
            for obs in self.obstacles:
                if (obs.x <= px <= obs.x + obs.width and
                        obs.y <= py <= obs.y + obs.height):
                    return dist
        return min_dist

    def _action_to_vel(self, action: int) -> tuple[float, float]:
        actions = [
            (self.config.max_speed, 0.0),
            (self.config.max_speed * 0.5, self.config.max_angular_speed * 0.5),
            (self.config.max_speed * 0.5, -self.config.max_angular_speed * 0.5),
            (0.0, self.config.max_angular_speed),
            (0.0, -self.config.max_angular_speed),
            (-self.config.max_speed * 0.3, 0.0),
            (self.config.max_speed * 0.7, self.config.max_angular_speed * 0.3),
            (self.config.max_speed * 0.7, -self.config.max_angular_speed * 0.3),
            (0.0, 0.0),
        ]
        return actions[action % len(actions)]

    def _check_collision(self):
        for obs in self.obstacles:
            closest_x = max(obs.x, min(self.robot.x, obs.x + obs.width))
            closest_y = max(obs.y, min(self.robot.y, obs.y + obs.height))
            dx = self.robot.x - closest_x
            dy = self.robot.y - closest_y
            if dx * dx + dy * dy < self.config.robot_radius ** 2:
                self.collided = True
                return

    def _check_bounds(self):
        r = self.config.robot_radius
        if (self.robot.x < r or self.robot.x > self.config.width - r or
                self.robot.y < r or self.robot.y > self.config.height - r):
            self.collided = True
        self.robot.x = max(r, min(self.config.width - r, self.robot.x))
        self.robot.y = max(r, min(self.config.height - r, self.robot.y))

    def _distance_to_goal(self) -> float:
        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y
        return math.sqrt(dx * dx + dy * dy)

    def _angle_to_goal(self) -> float:
        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y
        angle = math.atan2(dy, dx) - self.robot.heading
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _compute_reward(self, action: int) -> float:
        curr_dist = self._distance_to_goal()
        if curr_dist < self.config.goal_radius:
            self.reached_goal = True
            return 100.0 + max(0, (self.max_steps - self.steps)) * 0.5
        if self.collided:
            return -20.0
        prev_dist = getattr(self, '_prev_dist', curr_dist)
        self._prev_dist = curr_dist
        reward = (prev_dist - curr_dist) * 5.0
        min_lidar = min(r.distance for r in self.get_lidar()) if self.get_lidar() else 5.0
        if min_lidar < 1.0:
            reward -= 2.0
        if action == 8:
            reward -= 0.2
        reward -= 0.05
        return reward


def create_world(seed: int = None, width: float = 20.0, height: float = 20.0) -> World2D:
    config = SimConfig(width=width, height=height)
    return World2D(config=config, seed=seed)


def step_world(world: World2D, action: int) -> tuple[dict, float, bool, dict]:
    return world.step(action)


def reset_world(world: World2D) -> dict:
    return world.reset()


def render_world(world: World2D, save_path: str = None, show: bool = False) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, world.config.width)
    ax.set_ylim(0, world.config.height)
    ax.set_aspect('equal')
    ax.set_title(f"Step {world.steps} | Dist: {world._distance_to_goal():.1f}")

    for obs in world.obstacles:
        rect = patches.Rectangle((obs.x, obs.y), obs.width, obs.height,
                                  linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    ax.plot(world.goal[0], world.goal[1], 'g*', markersize=20, label='Goal')
    circle = patches.Circle((world.robot.x, world.robot.y), world.config.robot_radius,
                             color='blue', alpha=0.7)
    ax.add_patch(circle)
    ax.plot(world.robot.x, world.robot.y, 'bo', markersize=4)

    arrow_len = 0.8
    ax.arrow(world.robot.x, world.robot.y,
             arrow_len * math.cos(world.robot.heading),
             arrow_len * math.sin(world.robot.heading),
             head_width=0.2, head_length=0.15, fc='red', ec='red')

    lidar = world.get_lidar()
    for reading in lidar:
        end_x = world.robot.x + reading.distance * math.cos(reading.angle)
        end_y = world.robot.y + reading.distance * math.sin(reading.angle)
        color = 'red' if reading.hit else 'lightcoral'
        ax.plot([world.robot.x, end_x], [world.robot.y, end_y],
                color=color, linewidth=0.5, alpha=0.6)

    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
    if show:
        plt.show()
    plt.close()

    return save_path or ""


def evaluate_policy(world: World2D, policy_fn, num_episodes: int = 10) -> dict:
    results = {"successes": 0, "collisions": 0, "timeouts": 0, "total_reward": 0.0, "episodes": []}

    for ep in range(num_episodes):
        obs = world.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < world.max_steps:
            action = policy_fn(obs)
            obs, reward, done, info = world.step(action)
            episode_reward += reward
            steps += 1

        results["total_reward"] += episode_reward
        if info.get("reached_goal"):
            results["successes"] += 1
        elif info.get("collided"):
            results["collisions"] += 1
        else:
            results["timeouts"] += 1
        results["episodes"].append({
            "reward": episode_reward,
            "steps": steps,
            "success": info.get("reached_goal", False),
            "collision": info.get("collided", False),
        })

    results["success_rate"] = results["successes"] / num_episodes
    results["avg_reward"] = results["total_reward"] / num_episodes
    return results
