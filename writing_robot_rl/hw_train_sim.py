#!/usr/bin/env python3
"""Simulation-based RL training for the writing robot, calibrated to the real robot.

Trains fast in simulation (hundreds of steps/sec), using the same 5/4.8 steps-per-mm
calibration and safe bounds (X<=52mm, Y<=49mm) as the real hardware.
Usage:
    python3 hw_train_sim.py --timesteps 20000 [--render]
"""
import argparse, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import SimulatedHardware

# Calibrated bounds (mirror real robot)
MAX_X_MM = 52.0   # (275-15)/5.0
MAX_Y_MM = 49.0   # (250-15)/4.8
TRAJ = np.array([[5, 5], [5, 40], [35, 40], [35, 5]], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="model .zip to continue")
    args = ap.parse_args()

    env = WritingRobotEnv(
        trajectory=TRAJ,
        use_hardware=False,
        render_mode="human" if args.render else None,
        x_range=(0.0, MAX_X_MM),
        y_range=(0.0, MAX_Y_MM),
        max_step_mm=1.5,
        max_episode_steps=120,
        steps_per_mm=5.0,
        steps_per_mm_x=5.0,
        steps_per_mm_y=4.8,
        max_steps=(275, 250),
        safety_margin_steps=15,
        trajectory_density=2.0,
    )

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import ProgressBarCallback

    vec_env = DummyVecEnv([lambda: env])

    # bounds monitor callback
    from stable_baselines3.common.callbacks import BaseCallback
    class BoundsCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.max_x = self.max_y = 0.0
            self.viol = 0; self.steps = 0
            self.progress_sum = 0.0
            self.n_eps = 0
        def _on_step(self):
            pos = self.locals["new_obs"]["position"][0]
            self.steps += 1
            self.max_x = max(self.max_x, float(pos[0]))
            self.max_y = max(self.max_y, float(pos[1]))
            if float(pos[0]) > MAX_X_MM or float(pos[1]) > MAX_Y_MM:
                self.viol += 1
            return True
        def _on_rollout_start(self):
            self.roll_start = self.steps
        def _on_rollout_end(self):
            ep = self.model.get_vec_normalize_env()
            # report per rollout
            print(f"  sim steps={self.steps} maxX={self.max_x:.1f} maxY={self.max_y:.1f} viol={self.viol}")

    bcb = BoundsCB()
    use_norm = False
    if args.resume:
        model = PPO.load(args.resume, env=vec_env, verbose=1)
    else:
        model = PPO("MultiInputPolicy", vec_env, learning_rate=3e-4,
                    n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    vf_coef=0.5, max_grad_norm=0.5, verbose=1)

    t0 = time.time()
    print(f"Sim training {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=bcb, progress_bar=False,
                reset_num_timesteps=not bool(args.resume))
    dt = time.time() - t0
    model.save("writing_robot_sim_model")
    print(f"\nDONE in {dt:.0f}s ({args.timesteps/max(dt,0.001):.0f} fps)")
    print(f"Bounds: max X {bcb.max_x:.2f} (<=52) max Y {bcb.max_y:.2f} (<=49) violations {bcb.viol}")
    env.close()

if __name__ == "__main__":
    main()