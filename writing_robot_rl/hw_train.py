#!/usr/bin/env python3
"""
Quick hardware training — scaled to DVD sled travel (~30mm).
No pygame rendering — watch the real motors move.
"""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import HardwareInterface

# Physical sled travel (measured)
TRAVEL_X = 55.0   # 5.5 cm
TRAVEL_Y = 52.0   # 5.2 cm
MARGIN   = 5.0    # mm margin at each end (extra safety)

MAX_X = TRAVEL_X - MARGIN   # 50 mm
MAX_Y = TRAVEL_Y - MARGIN   # 47 mm

print(f"Safe workspace: X=0..{MAX_X}mm  Y=0..{MAX_Y}mm  (travel X=55,Y=52 minus {MARGIN}mm margin)")

# L-shape trajectory, vertical-first, fully inside safe workspace
trajectory = np.array([
    [5, 5], [5, 40], [35, 40], [35, 5]
], dtype=np.float32)

print("Hardware Training — DVD Writing Robot")
print(f"Workspace: 0..{MAX_X} x 0..{MAX_Y} mm")
print(f"Trajectory: {len(trajectory)} corner points (L-shape, vertical-first)")
print(f"Port: /dev/ttyACM0")
print()

# Quick hardware sanity check first
print("Testing connection...")
hw = HardwareInterface(port="/dev/ttyACM0", steps_per_mm=80.0,
                       physical_travel_mm=(TRAVEL_X, TRAVEL_Y),
                       safety_margin_mm=MARGIN)
hw.connect()
hw.set_speed(40)
hw.home()
print(f"  Home OK — position: {hw.get_position()}")
time.sleep(1)

# Move to corner 1 to verify movement
print("  Moving to (10, 10) mm to verify...")
hw.move_to_mm(10.0, 10.0)
time.sleep(2)
hw.home()
time.sleep(1)
print("  Hardware check OK — disconnecting test connection...")
hw.disconnect()
time.sleep(1)
print("  Training starts in 3s...")
print("  ⚠  The robot will move randomly during training.")
print("  ⚠  Watch it closely! Press Ctrl+C to stop.")
time.sleep(3)

# Create environment
env = WritingRobotEnv(
    trajectory=trajectory,
    use_hardware=True,
    port="/dev/ttyACM0",
    render_mode=None,        # no pygame — real motors ARE the visualization
    x_range=(0.0, MAX_X),
    y_range=(0.0, MAX_Y),
    max_step_mm=1.0,         # conservative step size for real hardware
    max_episode_steps=80,    # short episodes — hardware is slow
    steps_per_mm=80.0,
    trajectory_density=2.0,  # fewer waypoints for speed
)

# PPO training
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=1e-3,
        n_steps=16,           # small batch — hardware is slow
        batch_size=16,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
    )

    print("\nTraining started (Ctrl+C to stop and save)...")
    model.learn(total_timesteps=512)  # 512 steps to start
    model.save("writing_robot_hw_model")
    print("Saved: writing_robot_hw_model.zip")

except KeyboardInterrupt:
    print("\nInterrupted — returning to home...")
finally:
    print("Homing motors...")
    env.reset()
    env.close()
    print("Done.")
