"""Writing Robot RL Package"""
from .env import WritingRobotEnv
from .hardware import HardwareInterface, SimulatedHardware
from .trajectory import (
    svg_to_trajectory, 
    interpolate_trajectory, 
    trajectory_from_text,
    smooth_trajectory,
    resample_trajectory,
    generate_basic_shapes,
)
from .reward import (
    compute_reward,
    dense_reward,
    shaped_reward,
    compute_reward_components,
    curriculum_reward,
)
from .sim import (
    Viewer2D,
    create_viewer,
    MeshCatViewer,
    ThreeJSViewer,
    create_3d_viewer,
)

__all__ = [
    # Core
    "WritingRobotEnv",
    "HardwareInterface",
    "SimulatedHardware",
    # Trajectory
    "svg_to_trajectory",
    "interpolate_trajectory",
    "trajectory_from_text",
    "smooth_trajectory",
    "resample_trajectory",
    "generate_basic_shapes",
    # Reward
    "compute_reward",
    "dense_reward",
    "shaped_reward",
    "compute_reward_components",
    "curriculum_reward",
    # Visualization
    "Viewer2D",
    "create_viewer",
    "MeshCatViewer",
    "ThreeJSViewer",
    "create_3d_viewer",
]