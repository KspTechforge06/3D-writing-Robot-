"""
Reward functions for Writing Robot RL.
"""
import numpy as np
from typing import Dict, Any, Optional


def compute_reward(
    prev_obs: Dict[str, np.ndarray],
    action: np.ndarray,
    obs: Dict[str, np.ndarray],
    target_path: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute step reward for writing robot.
    
    Components:
    - Path following: negative distance to closest path point
    - Progress: increase in path_progress
    - Smoothness: penalize acceleration
    - Pen state: match target pen state
    - Completion: bonus for finishing trajectory
    """
    if weights is None:
        weights = {
            "path": 10.0,
            "progress": 100.0,
            "smooth": 0.1,
            "pen": 1.0,
            "completion": 50.0,
        }
    
    # Path following reward (negative distance to path)
    pos = obs["position"]
    distances = np.linalg.norm(target_path - pos, axis=1)
    min_dist = np.min(distances)
    path_reward = -min_dist * weights["path"]
    
    # Progress reward
    prev_progress = prev_obs.get("path_progress", np.array([0.0]))[0]
    curr_progress = obs.get("path_progress", np.array([0.0]))[0]
    progress_reward = (curr_progress - prev_progress) * weights["progress"]
    
    # Smoothness penalty (acceleration)
    prev_vel = prev_obs.get("velocity", np.array([0.0, 0.0]))
    curr_vel = obs.get("velocity", np.array([0.0, 0.0]))
    accel = np.linalg.norm(curr_vel - prev_vel)
    smooth_reward = -accel * weights["smooth"]
    
    # Pen state reward (if target pen state available)
    pen_reward = 0.0
    if "target_pen_state" in obs:
        pen_match = 1.0 if obs["pen_state"][0] == obs["target_pen_state"][0] else -1.0
        pen_reward = pen_match * weights["pen"]
    
    # Completion bonus
    completion_reward = 0.0
    if obs.get("path_progress", np.array([0.0]))[0] >= 1.0:
        completion_reward = weights["completion"]
    
    total = (
        path_reward + 
        progress_reward + 
        smooth_reward + 
        pen_reward + 
        completion_reward
    )
    
    return float(total)


def dense_reward(
    obs: Dict[str, np.ndarray],
    target_path: np.ndarray,
    target_pen_state: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Dense reward based only on current observation (no prev_obs needed).
    Useful for debugging or when prev_obs not available.
    """
    if weights is None:
        weights = {"path": 10.0, "pen": 1.0, "completion": 50.0}
    
    pos = obs["position"]
    distances = np.linalg.norm(target_path - pos, axis=1)
    min_dist = np.min(distances)
    
    path_reward = -min_dist * weights["path"]
    
    pen_reward = 0.0
    if target_pen_state is not None:
        pen_match = 1.0 if obs["pen_state"][0] == target_pen_state else -1.0
        pen_reward = pen_match * weights["pen"]
    
    completion = 0.0
    if obs.get("path_progress", np.array([0.0]))[0] >= 1.0:
        completion = weights["completion"]
    
    return float(path_reward + pen_reward + completion)


def shaped_reward(
    prev_obs: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    target_path: np.ndarray,
    target_pen_state: Optional[float] = None,
    potential_fn=None,
) -> float:
    """
    Potential-based reward shaping (Ng et al. 1999).
    Guarantees policy invariance.
    
    F = γΦ(s') - Φ(s)
    where Φ is a potential function (e.g., negative distance to goal).
    """
    if potential_fn is None:
        # Default potential: negative distance to path + progress
        def potential_fn(o):
            pos = o["position"]
            dist = np.min(np.linalg.norm(target_path - pos, axis=1))
            prog = o.get("path_progress", np.array([0.0]))[0]
            return -dist + 10.0 * prog
    
    gamma = 0.99
    phi_prev = potential_fn(prev_obs)
    phi_curr = potential_fn(obs)
    
    return float(gamma * phi_curr - phi_prev)


def compute_reward_components(
    prev_obs: Dict[str, np.ndarray],
    action: np.ndarray,
    obs: Dict[str, np.ndarray],
    target_path: np.ndarray,
) -> Dict[str, float]:
    """Return individual reward components for logging/debugging."""
    pos = obs["position"]
    distances = np.linalg.norm(target_path - pos, axis=1)
    min_dist = np.min(distances)
    
    prev_progress = prev_obs.get("path_progress", np.array([0.0]))[0]
    curr_progress = obs.get("path_progress", np.array([0.0]))[0]
    
    prev_vel = prev_obs.get("velocity", np.array([0.0, 0.0]))
    curr_vel = obs.get("velocity", np.array([0.0, 0.0]))
    accel = np.linalg.norm(curr_vel - prev_vel)
    
    return {
        "path_dist": float(min_dist),
        "path_reward": -min_dist * 10.0,
        "progress_delta": float(curr_progress - prev_progress),
        "progress_reward": (curr_progress - prev_progress) * 100.0,
        "accel": float(accel),
        "smooth_reward": -accel * 0.1,
        "pen_state": float(obs["pen_state"][0]),
    }


def curriculum_reward(
    base_reward: float,
    episode: int,
    total_episodes: int,
    curriculum_type: str = "linear",
) -> float:
    """
    Curriculum learning: adjust reward scale during training.
    
    Early training: emphasize exploration, reduce completion bonus
    Late training: emphasize precision, increase completion bonus
    """
    progress = episode / max(1, total_episodes)
    
    if curriculum_type == "linear":
        # Linearly increase completion weight
        completion_scale = 0.1 + 0.9 * progress
    elif curriculum_type == "cosine":
        import math
        completion_scale = 0.1 + 0.9 * (1 - math.cos(progress * math.pi)) / 2
    elif curriculum_type == "step":
        completion_scale = 0.1 if progress < 0.5 else 1.0
    else:
        completion_scale = 1.0
    
    return base_reward * completion_scale