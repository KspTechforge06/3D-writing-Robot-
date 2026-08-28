"""
Gymnasium Environment for 2-Axis Writing Robot

Supports both simulation (fast, no hardware) and real hardware modes.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import time
import math

from .hardware import HardwareInterface
from .trajectory import interpolate_trajectory
from .reward import compute_reward


class WritingRobotEnv(gym.Env):
    """
    2-axis writing robot environment.
    
    Observation:
        Dict with keys:
        - position: (2,) current XY in mm
        - velocity: (2,) current velocity mm/step
        - pen_state: (1,) 0=up, 1=down
        - target_path: (N, 2) waypoints in mm
        - path_progress: (1,) 0.0 to 1.0
        - steps_remaining: (1,) int
    
    Action (continuous):
        Box(-1, 1) shape (3,) -> [dx, dy, pen]
        dx, dy: normalized step (-1 to 1 maps to max_step_mm)
        pen: >0 = down, <=0 = up
    
    Reward:
        - Path following (distance to trajectory)
        - Progress along path
        - Smoothness (acceleration penalty)
        - Pen state correctness
        - Completion bonus
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        # Workspace
        x_range: Tuple[float, float] = (0.0, 100.0),  # mm
        y_range: Tuple[float, float] = (0.0, 100.0),  # mm
        # Motion
        max_step_mm: float = 2.0,          # max mm per action step
        max_velocity_mm: float = 10.0,     # mm/step (for obs normalization)
        max_accel_mm: float = 5.0,         # mm/step^2
        # Episode
        max_episode_steps: int = 500,
        # Trajectory
        trajectory: Optional[np.ndarray] = None,  # (N, 2) waypoints
        trajectory_density: float = 1.0,   # mm between interpolated points
        # Pen
        pen_servo_channel: int = 1,        # servo channel (1 or 2)
        pen_up_angle: int = 90,
        pen_down_angle: int = 45,
        # Hardware
        use_hardware: bool = False,
        port: Optional[str] = None,
        baud: int = 9600,
        steps_per_mm: float = 5.0,
        steps_per_mm_x: Optional[float] = None,
        steps_per_mm_y: Optional[float] = None,
        max_steps: Tuple[int, int] = (275, 250),
        safety_margin_steps: int = 15,
        # Rendering
        render_mode: Optional[str] = None,
        window_size: Tuple[int, int] = (800, 600),
        # Reward weights
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Store config
        self.x_range = x_range
        self.y_range = y_range
        self.max_step_mm = max_step_mm
        self.max_velocity_mm = max_velocity_mm
        self.max_accel_mm = max_accel_mm
        self.max_episode_steps = max_episode_steps
        self.pen_servo_channel = pen_servo_channel
        self.pen_up_angle = pen_up_angle
        self.pen_down_angle = pen_down_angle
        self.use_hardware = use_hardware
        self.render_mode = render_mode
        self.window_size = window_size
        
        # Default trajectory (square) if none provided
        if trajectory is None:
            trajectory = np.array([
                [10, 10], [90, 10], [90, 90], [10, 90], [10, 10]
            ], dtype=np.float32)
        self.base_trajectory = trajectory.astype(np.float32)
        self.trajectory_density = trajectory_density
        
        # Reward weights
        self.reward_weights = {
            "path": 10.0,
            "progress": 100.0,
            "smooth": 0.1,
            "pen": 1.0,
            "completion": 50.0,
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)
        
        # Interpolate trajectory to dense waypoints
        self.target_path = interpolate_trajectory(
            self.base_trajectory, 
            density=self.trajectory_density
        )
        self.path_length = len(self.target_path)
        
        # Hardware interface
        self.hardware: Optional[HardwareInterface] = None
        if use_hardware:
            self.hardware = HardwareInterface(
                port=port, baud=baud, steps_per_mm=steps_per_mm,
                steps_per_mm_x=steps_per_mm_x,
                steps_per_mm_y=steps_per_mm_y,
                max_steps=max_steps,
                safety_margin_steps=safety_margin_steps,
            )
            self.hardware.connect()
        
        # State variables
        self.position = np.array([x_range[0], y_range[0]], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.pen_state = 0.0
        self.path_progress = 0.0
        self.step_count = 0
        self.prev_distance_to_path = 0.0
        
        # Rendering
        self.viewer = None
        self._init_viewer()
        
        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space (Dict - matches _get_obs output)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([x_range[0], y_range[0]], dtype=np.float32),
                high=np.array([x_range[1], y_range[1]], dtype=np.float32),
                dtype=np.float32,
            ),
            "velocity": spaces.Box(
                low=-max_velocity_mm, high=max_velocity_mm, shape=(2,), dtype=np.float32
            ),
            "pen_state": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "path_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "steps_remaining": spaces.Box(
                0.0, float(max_episode_steps), shape=(1,), dtype=np.float32
            ),
            "current_target": spaces.Box(
                low=np.array([x_range[0], y_range[0]], dtype=np.float32),
                high=np.array([x_range[1], y_range[1]], dtype=np.float32),
                dtype=np.float32,
            ),
        })
        
        # For rendering
        self._last_render_time = 0.0
    
    def _init_viewer(self):
        if self.render_mode == "human":
            try:
                from .sim.viewer_2d import Viewer2D
                self.viewer = Viewer2D(
                    window_size=self.window_size,
                    x_range=self.x_range,
                    y_range=self.y_range,
                    target_path=self.target_path,
                )
            except ImportError:
                print("Warning: pygame not available, rendering disabled")
                self.render_mode = None
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        # Find closest point on target path
        distances = np.linalg.norm(self.target_path - self.position, axis=1)
        closest_idx = np.argmin(distances)
        self.path_progress = closest_idx / max(1, self.path_length - 1)
        
        # Current target waypoint (lookahead)
        target_idx = min(closest_idx + 5, self.path_length - 1)
        current_target = self.target_path[target_idx]
        
        return {
            "position": self.position.copy().astype(np.float32),
            "velocity": self.velocity.copy().astype(np.float32),
            "pen_state": np.array([self.pen_state], dtype=np.float32),
            "path_progress": np.array([self.path_progress], dtype=np.float32),
            "steps_remaining": np.array(
                [self.max_episode_steps - self.step_count], dtype=np.float32
            ),
            "current_target": current_target.astype(np.float32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "position": self.position.copy(),
            "path_progress": self.path_progress,
            "step_count": self.step_count,
        }
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Reset to start of trajectory
        self.position = self.target_path[0].copy()
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.pen_state = 1.0  # start with pen down
        self.path_progress = 0.0
        self.step_count = 0
        self.prev_distance_to_path = 0.0
        
        # Hardware homing
        if self.use_hardware and self.hardware:
            self.hardware.home()
            self.hardware.set_pen(self.pen_down_angle)
            # Update position from hardware
            hw_pos = self.hardware.get_position()
            if hw_pos is not None:
                self.position = np.array(hw_pos, dtype=np.float32)
        
        # Pen down at start
        if self.use_hardware and self.hardware:
            self.hardware.set_pen(self.pen_down_angle)
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        dx, dy, pen_cmd = action
        
        # Scale action to mm
        step_x = np.clip(dx * self.max_step_mm, -self.max_step_mm, self.max_step_mm)
        step_y = np.clip(dy * self.max_step_mm, -self.max_step_mm, self.max_step_mm)
        
        # New position
        new_pos = self.position + np.array([step_x, step_y], dtype=np.float32)
        new_pos = np.clip(new_pos, [self.x_range[0], self.y_range[0]], 
                          [self.x_range[1], self.y_range[1]])
        
        # New velocity
        new_velocity = new_pos - self.position
        
        # Pen state
        new_pen = 1.0 if pen_cmd > 0 else 0.0
        
        # Store previous state for reward
        prev_obs = self._get_obs()
        
        # Update state
        self.position = new_pos
        self.velocity = new_velocity
        self.pen_state = new_pen
        self.step_count += 1
        
        # Hardware execution
        if self.use_hardware and self.hardware:
            self.hardware.move_to_mm(self.position[0], self.position[1])
            if self.pen_state != prev_obs["pen_state"][0]:
                angle = self.pen_down_angle if self.pen_state > 0.5 else self.pen_up_angle
                self.hardware.set_pen(angle)
        
        # Compute reward
        new_obs = self._get_obs()
        reward = compute_reward(
            prev_obs, action, new_obs, 
            self.target_path, 
            self.reward_weights
        )
        
        # Termination conditions
        terminated = self.path_progress >= 1.0  # completed trajectory
        truncated = self.step_count >= self.max_episode_steps
        
        info = self._get_info()
        info["reward_components"] = {
            "path": -np.linalg.norm(self.position - self._closest_path_point()),
            "progress": self.path_progress,
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return new_obs, reward, terminated, truncated, info
    
    def _closest_path_point(self) -> np.ndarray:
        distances = np.linalg.norm(self.target_path - self.position, axis=1)
        return self.target_path[np.argmin(distances)]
    
    def _render_frame(self):
        if self.viewer is None:
            self._init_viewer()
        if self.viewer:
            self.viewer.render(
                position=self.position,
                target_path=self.target_path,
                pen_state=self.pen_state,
                path_progress=self.path_progress,
            )
    
    def render(self):
        if self.render_mode == "rgb_array":
            # Return RGB array (implement in viewer)
            if self.viewer:
                return self.viewer.get_rgb_array()
            return np.zeros((*self.window_size[::-1], 3), dtype=np.uint8)
        elif self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_ansi(self) -> str:
        # Simple text rendering for terminal
        grid_w, grid_h = 40, 20
        grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]
        
        # Draw trajectory
        for pt in self.target_path:
            x = int((pt[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (grid_w - 1))
            y = int((pt[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (grid_h - 1))
            if 0 <= x < grid_w and 0 <= y < grid_h:
                grid[grid_h - 1 - y][x] = "."
        
        # Draw robot
        rx = int((self.position[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (grid_w - 1))
        ry = int((self.position[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (grid_h - 1))
        if 0 <= rx < grid_w and 0 <= ry < grid_h:
            grid[grid_h - 1 - ry][rx] = "█" if self.pen_state > 0.5 else "○"
        
        lines = ["".join(row) for row in grid]
        return "\n".join(lines) + f"\nPos: {self.position:.1f}mm  Progress: {self.path_progress:.1%}  Step: {self.step_count}"
    
    def close(self):
        if self.use_hardware and self.hardware:
            self.hardware.disconnect()
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def make_env(**kwargs) -> WritingRobotEnv:
    """Factory function for easy environment creation."""
    return WritingRobotEnv(**kwargs)