#!/usr/bin/env python3
"""
Short bounded test — 1 PPO batch (16 steps) then 4 more batches (~80 steps total).
Logs every commanded position and reports max reached vs safe limits.
"""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv

from writing_robot.hardware import HardwareInterface

# Track commanded moves at the hardware layer (ground truth for bounds)
_HW_MOVES = {"n": 0, "max_x": 0.0, "max_y": 0.0, "viol": 0}
_orig_move = HardwareInterface.move_to_mm
def _tracked_move(self, x_mm, y_mm):
    r = _orig_move(self, x_mm, y_mm)
    _HW_MOVES["n"] += 1
    _HW_MOVES["max_x"] = max(_HW_MOVES["max_x"], self.x_pos)
    _HW_MOVES["max_y"] = max(_HW_MOVES["max_y"], self.y_pos)
    if self.x_pos > self.limit_hi[0] or self.y_pos > self.limit_hi[1]:
        _HW_MOVES["viol"] += 1
    return r
HardwareInterface.move_to_mm = _tracked_move

TRAVEL_X = 55.0
TRAVEL_Y = 52.0
MARGIN   = 5.0
MAX_X = TRAVEL_X - MARGIN   # 50 mm
MAX_Y = TRAVEL_Y - MARGIN   # 47 mm

trajectory = np.array([
    [5, 5], [5, 40], [35, 40], [35, 5]
], dtype=np.float32)

print(f"Safe limits: X<= {MAX_X}mm, Y<= {MAX_Y}mm")

env = WritingRobotEnv(
    trajectory=trajectory,
    use_hardware=True,
    port="/dev/ttyACM0",
    render_mode=None,
    x_range=(0.0, MAX_X),
    y_range=(0.0, MAX_Y),
    max_step_mm=1.0,
    max_episode_steps=80,
    steps_per_mm=80.0,
    physical_travel_mm=(TRAVEL_X, TRAVEL_Y),
    safety_margin_mm=MARGIN,
    trajectory_density=2.0,
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class BoundsTracker(BaseCallback):
    def __init__(self, max_x, max_y):
        super().__init__()
        self.max_x, self.max_y = max_x, max_y
        self.max_reached_x = 0.0
        self.max_reached_y = 0.0
        self.violations = 0
        self.n = 0
    def _on_step(self):
        pos = self.locals["new_obs"]["position"]
        for p in pos:
            x, y = float(p[0]), float(p[1])
            self.n += 1
            self.max_reached_x = max(self.max_reached_x, x)
            self.max_reached_y = max(self.max_reached_y, y)
            if x > self.max_x or y > self.max_y or x < 0 or y < 0:
                self.violations += 1

tracker = BoundsTracker(MAX_X, MAX_Y)

vec_env = DummyVecEnv([lambda: env])
model = PPO(
    "MultiInputPolicy",
    vec_env,
    learning_rate=1e-3,
    n_steps=16,
    batch_size=16,
    n_epochs=5,
    gamma=0.99,
    verbose=0,
)

print("\nRunning short bounded test...")
model.learn(total_timesteps=80, callback=tracker)
model.save("writing_robot_hw_model")
env.close()

print("\n=== BOUNDS REPORT ===")
print(f"Hardware moves        : {_HW_MOVES['n']}")
print(f"Hardware max X reached: {_HW_MOVES['max_x']:.2f}mm  (limit {MAX_X}mm)")
print(f"Hardware max Y reached: {_HW_MOVES['max_y']:.2f}mm  (limit {MAX_Y}mm)")
print(f"Hardware violations   : {_HW_MOVES['viol']}")
print(f"Steps executed (obs)  : {tracker.n}")
print(f"Max X commanded (obs) : {tracker.max_reached_x:.2f}mm  (limit {MAX_X}mm)")
print(f"Max Y commanded (obs) : {tracker.max_reached_y:.2f}mm  (limit {MAX_Y}mm)")
print(f"Boundary violations   : {tracker.violations}")
if _HW_MOVES["viol"] == 0 and tracker.violations == 0:
    print("RESULT: STAYED IN BOUNDS ✓")
else:
    print("RESULT: OVERFLOW ✗")