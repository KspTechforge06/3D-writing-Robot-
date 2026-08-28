#!/usr/bin/env python3
"""
Quick test to verify the Writing Robot RL environment works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from writing_robot.env import WritingRobotEnv
from writing_robot.trajectory import generate_basic_shapes


def test_env():
    print("Testing WritingRobotEnv...")
    
    # Create env with square trajectory
    shapes = generate_basic_shapes()
    trajectory = shapes["square"]
    
    env = WritingRobotEnv(
        trajectory=trajectory,
        use_hardware=False,
        render_mode="human",
        max_episode_steps=200,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset
    obs, info = env.reset()
    print(f"Initial obs keys: {obs.keys()}")
    print(f"Initial position: {obs['position']}")
    
    # Run a few steps
    total_reward = 0
    for step in range(50):
        # Simple policy: move toward current target
        target = obs["current_target"]
        pos = obs["position"]
        error = target - pos
        dist = np.linalg.norm(error)
        
        if dist < 0.5:
            action = np.array([0.0, 0.0, 1.0])
        else:
            action = np.array([error[0]/10.0, error[1]/10.0, 1.0])
        
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: pos={obs['position']}, target={obs['current_target']}, "
                  f"progress={obs['path_progress'][0]:.2%}, reward={reward:.2f}")
        
        if terminated:
            print(f"Completed! Progress: {info['path_progress']:.1%}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    
    # Test reset
    obs, info = env.reset()
    print(f"After reset: pos={obs['position']}")
    
    env.close()
    print("Test passed!")


def test_trajectory():
    print("\nTesting trajectory utilities...")
    from writing_robot.trajectory import interpolate_trajectory, generate_basic_shapes
    
    shapes = generate_basic_shapes()
    for name, traj in shapes.items():
        print(f"  {name}: {len(traj)} points, range X[{traj[:,0].min():.1f}-{traj[:,0].max():.1f}] Y[{traj[:,1].min():.1f}-{traj[:,1].max():.1f}]")
    
    # Test interpolation
    sparse = np.array([[0,0], [10,0], [10,10]], dtype=np.float32)
    dense = interpolate_trajectory(sparse, density=1.0)
    print(f"  Sparse (3 pts) -> Dense ({len(dense)} pts)")
    
    print("Trajectory tests passed!")


def test_hardware_sim():
    print("\nTesting simulated hardware...")
    from writing_robot.hardware import SimulatedHardware
    
    hw = SimulatedHardware()
    hw.connect()
    
    hw.move_to_mm(10.0, 20.0)
    pos = hw.get_position()
    print(f"  Position after move: {pos}")
    
    hw.home()
    pos = hw.get_position()
    print(f"  Position after home: {pos}")
    
    hw.disconnect()
    print("Simulated hardware test passed!")


if __name__ == "__main__":
    test_trajectory()
    test_hardware_sim()
    test_env()
    print("\n✓ All tests passed!")