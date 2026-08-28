#!/usr/bin/env python3
"""
Round-based RL training on real hardware.
One PPO batch per round, then a report (progress, reward, bounds check).
Weights can be changed between rounds via --weights "path=10,progress=100,...".

Usage:
  python3 hw_round.py --round 1 --timesteps 48                      # fresh model, default weights
  python3 hw_round.py --round 2 --timesteps 48 --weights "path=2,progress=200,smooth=0.5,completion=0"
  python3 hw_round.py --round 3 --resume                             # keep last weights, continue
"""
import sys, time, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import HardwareInterface

# ---- physical robot constants (measured) ----
TRAVEL_X = 275          # steps to dead end
TRAVEL_Y = 250          # steps to dead end
STEPS_X  = 5.0          # steps/mm (X)
STEPS_Y  = 4.8          # steps/mm (Y)
MARGIN   = 15           # safety steps from each dead end

MAX_X_MM = (TRAVEL_X - MARGIN) / STEPS_X   # 52.0 mm
MAX_Y_MM = (TRAVEL_Y - MARGIN) / STEPS_Y   # ~48.96 mm
print(f"SAFE BOUNDS: X 0..{MAX_X_MM:.1f}mm ({TRAVEL_X-MARGIN} steps)  Y 0..{MAX_Y_MM:.1f}mm ({TRAVEL_Y-MARGIN} steps)")

# L-shape trajectory inside bounds
trajectory = np.array([
    [5, 5], [5, 40], [35, 40], [35, 5]
], dtype=np.float32)

WEIGHTS_FILE = Path(__file__).parent / "hw_weights.json"
MODEL_FILE   = Path(__file__).parent / "writing_robot_hw_model"

def load_weights():
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text())
    return {"path": 10.0, "progress": 100.0, "smooth": 0.1, "pen": 1.0, "completion": 50.0}

def boundary_report(vec_env):
    """Read current hardware position; report max reached by reading pos each step."""
    # Track via a lightweight wrapper on the env's position
    env = vec_env.envs[0]
    return env.position

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=48)
    ap.add_argument("--weights", type=str, default=None,
                    help='e.g. "path=2,progress=200,smooth=0.5,completion=0"')
    ap.add_argument("--resume", action="store_true",
                    help="continue from saved model (default: only if exists)")
    ap.add_argument("--log-steps", action="store_true",
                    help="print every commanded position while running")
    args = ap.parse_args()

    # ---- weights ----
    weights = load_weights()
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v.strip())
    WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))
    print(f"\nROUND {args.round}  |  weights: {weights}")

    # ---- env ----
    env = WritingRobotEnv(
        trajectory=trajectory,
        use_hardware=True,
        port="/dev/ttyACM0",
        render_mode=None,
        x_range=(0.0, MAX_X_MM),
        y_range=(0.0, MAX_Y_MM),
        max_step_mm=1.5,
        max_episode_steps=60,
        steps_per_mm=5.0,
        steps_per_mm_x=STEPS_X,
        steps_per_mm_y=STEPS_Y,
        max_steps=(TRAVEL_X, TRAVEL_Y),
        safety_margin_steps=MARGIN,
        reward_weights=weights,
        trajectory_density=2.0,
    )

    # ---- bounds monitor (clips obs position) ----
    import stable_baselines3.common.callbacks as cb

    class BoundsCB(cb.BaseCallback):
        def __init__(self):
            super().__init__()
            self.max_x = 0.0; self.max_y = 0.0
            self.viol = 0; self.moves = 0
        def _on_step(self):
            pos = self.locals["new_obs"]["position"]
            x, y = float(pos[0][0]), float(pos[0][1])
            self.moves += 1
            self.max_x = max(self.max_x, x)
            self.max_y = max(self.max_y, y)
            if x > MAX_X_MM or y > MAX_Y_MM or x < 0 or y < 0:
                self.viol += 1
            return True

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])

    model_path = str(MODEL_FILE) + ".zip"
    if args.resume or Path(model_path).exists():
        print("  resuming:", model_path)
        model = PPO.load(model_path, env=vec_env, verbose=0)
        # reflect new weights (env object already updated)
    else:
        print("  new model (random init)")
        model = PPO(
            "MultiInputPolicy", vec_env,
            learning_rate=5e-3,
            n_steps=args.timesteps if args.timesteps >= 8 else 8,
            batch_size=8,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
        )

    # ensure the model's env uses the *same* dict obs (it will, MultiInputPolicy)
    bcb = BoundsCB()
    print(f"  training {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=bcb, reset_num_timesteps=(args.round == 1))
    model.save(str(MODEL_FILE))

    env2 = vec_env.envs[0]
    print("\n===== REPORT (round %d) =====" % args.round)
    print(f"weight path      : {weights.get('path')}")
    print(f"weight progress  : {weights.get('progress')}")
    print(f"weight smooth    : {weights.get('smooth')}")
    print(f"weight pen       : {weights.get('pen')}")
    print(f"weight completion: {weights.get('completion')}")
    print(f"moves             : {bcb.moves}")
    print(f"max X commanded   : {bcb.max_x:.2f}mm  (bound {MAX_X_MM:.1f})")
    print(f"max Y commanded   : {bcb.max_y:.2f}mm  (bound {MAX_Y_MM:.1f})")
    print(f"boundary violations: {bcb.viol}")
    print(f"final position    : {env2.position}")

    # reward breakdown
    total_reward = 0.0
    # recompute from last obs is complex; show env progress
    print(f"path_progress     : {env2.path_progress:.2%}")
    print(f"distance to path  : {np.linalg.norm(env2.position - env2._closest_path_point()):.2f}mm")
    print(f"steps             : {env2.step_count}")
    if bcb.viol == 0:
        print("BOUNDS: OK ✓")
    else:
        print("BOUNDS: VIOLATION ✗")

    env.close()

if __name__ == "__main__":
    main()