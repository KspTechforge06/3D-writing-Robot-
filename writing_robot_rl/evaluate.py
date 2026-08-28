#!/usr/bin/env python3
"""
Evaluation script for trained Writing Robot models.
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.trajectory import generate_basic_shapes


def evaluate_manual_control(env):
    """Interactive manual control for testing."""
    print("Manual control mode - use keyboard:")
    print("  WASD / Arrow keys: Move")
    print("  SPACE: Toggle pen")
    print("  R: Reset to home")
    print("  Q: Quit")
    
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Manual Control - Focus this window")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    
    # Action state
    action = [0.0, 0.0, 1.0]  # dx, dy, pen
    step_size = 2.0  # mm per key press
    
    obs, _ = env.reset()
    total_reward = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0
                elif event.key == pygame.K_SPACE:
                    action[2] = 1.0 if action[2] <= 0 else -1.0
                elif event.key in (pygame.K_w, pygame.K_UP):
                    action[1] = 1.0
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    action[1] = -1.0
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    action[0] = -1.0
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    action[0] = 1.0
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN):
                    action[1] = 0.0
                if event.key in (pygame.K_a, pygame.K_d, pygame.K_LEFT, pygame.K_RIGHT):
                    action[0] = 0.0
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        
        # Draw UI
        screen.fill((30, 30, 40))
        font = pygame.font.SysFont("monospace", 16)
        lines = [
            f"Pos: ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}) mm",
            f"Progress: {obs['path_progress'][0]:.1%}",
            f"Pen: {'DOWN' if obs['pen_state'][0] > 0.5 else 'UP'}",
            f"Reward: {reward:.2f} | Total: {total_reward:.1f}",
            f"Step: {info['step_count']}",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (255, 255, 255))
            screen.blit(surf, (10, 10 + i * 25))
        
        help_text = "WASD/Arrows: Move | SPACE: Pen | R: Reset | Q: Quit"
        surf = pygame.font.SysFont("monospace", 12).render(help_text, True, (150, 150, 150))
        screen.blit(surf, (10, 160))
        
        pygame.display.flip()
        pygame.time.Clock().tick(30)
        
        if terminated:
            print(f"Episode done! Progress: {info['path_progress']:.1%}")
            obs, _ = env.reset()
            total_reward = 0
    
    pygame.quit()


def benchmark_environment(env, n_steps: int = 1000):
    """Benchmark environment step speed."""
    import time
    import numpy as np
    
    env.reset()
    actions = np.random.uniform(-1, 1, (n_steps, 3)).astype(np.float32)
    
    start = time.time()
    for i in range(n_steps):
        env.step(actions[i])
    elapsed = time.time() - start
    
    print(f"Benchmark: {n_steps} steps in {elapsed:.3f}s = {n_steps/elapsed:.0f} steps/sec")
    print(f"Step latency: {elapsed/n_steps*1000:.3f} ms")


def test_trajectory_following(env, trajectory, n_episodes: int = 3):
    """Test if environment can follow a trajectory."""
    print(f"Testing trajectory following ({len(trajectory)} waypoints)...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        # Simple P-controller to follow path
        while not done:
            target = obs["current_target"]
            pos = obs["position"]
            error = target - pos
            dist = np.linalg.norm(error)
            
            if dist < 1e-3:
                action = np.array([0, 0, 1])  # pen down
            else:
                # Normalize and scale
                action_xy = error / (dist + 1e-6) * 0.5
                action = np.array([action_xy[0], action_xy[1], 1.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
        
        print(f"Episode {ep+1}: Progress {info['path_progress']:.1%}, Steps: {info['step_count']}")


def main():
    parser = argparse.ArgumentParser(description="Writing Robot Evaluation Tools")
    parser.add_argument("--mode", choices=["manual", "benchmark", "follow"], default="manual")
    parser.add_argument("--env", choices=["sim", "real"], default="sim")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--shape", type=str, default="square")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    
    shapes = generate_basic_shapes()
    trajectory = shapes.get(args.shape, shapes["square"])
    
    env = WritingRobotEnv(
        trajectory=trajectory,
        use_hardware=(args.env == "real"),
        port=args.port,
        render_mode="human",
    )
    
    try:
        if args.mode == "manual":
            evaluate_manual_control(env)
        elif args.mode == "benchmark":
            benchmark_environment(env, args.steps)
        elif args.mode == "follow":
            test_trajectory_following(env, trajectory, args.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()