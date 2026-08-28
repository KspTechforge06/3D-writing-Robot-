#!/usr/bin/env python3
"""
Training script for Writing Robot RL.

Usage:
    python -m writing_robot.train --env sim --algo ppo --timesteps 100000
    python -m writing_robot.train --env real --port /dev/ttyACM0 --model best_model.zip
"""
import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.trajectory import generate_basic_shapes, interpolate_trajectory


def make_env(env_type: str, **kwargs):
    """Create environment based on type."""
    kwargs.setdefault("render_mode", "human")
    if env_type == "sim":
        return WritingRobotEnv(use_hardware=False, **kwargs)
    elif env_type == "real":
        return WritingRobotEnv(use_hardware=True, **kwargs)
    else:
        raise ValueError(f"Unknown env type: {env_type}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_ppo(
    env,
    total_timesteps: int = 100000,
    config: Optional[dict] = None,
    model_path: Optional[str] = None,
):
    """Train PPO using Stable Baselines3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3")
        return
    
    # Default PPO config
    import gymnasium as gym_spaces
    policy = (
        "MultiInputPolicy"
        if isinstance(env.observation_space, gym_spaces.spaces.Dict)
        else "MlpPolicy"
    )
    ppo_config = {
        "policy": policy,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }
    if config:
        ppo_config.update(config.get("ppo", {}))
    
    # Vectorize env
    vec_env = DummyVecEnv([lambda: env])
    if config and config.get("normalize", True):
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Model
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=vec_env)
        print(f"Loaded model from {model_path}")
    else:
        model = PPO(env=vec_env, **ppo_config)
    
    # Callbacks
    callbacks = []
    if config and config.get("checkpoint_freq", 0) > 0:
        callbacks.append(CheckpointCallback(
            save_freq=config["checkpoint_freq"],
            save_path="./checkpoints/",
            name_prefix="writing_robot",
        ))
    
    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # Save final model
    save_path = "writing_robot_ppo_final.zip"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    if isinstance(vec_env, VecNormalize):
        vec_env.save("vec_normalize.pkl")
    
    return model


def train_custom(env, total_timesteps: int, config: dict):
    """Custom training loop (no SB3 dependency)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    
    print("Using custom PPO implementation (no SB3)...")
    
    # Simple Actor-Critic network
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.actor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim),
                nn.Tanh(),
            )
            self.log_std = nn.Parameter(torch.zeros(act_dim))
            self.critic = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        
        def forward(self, obs):
            features = self.shared(obs)
            mean = self.actor(features)
            std = self.log_std.exp()
            value = self.critic(features)
            return mean, std, value
        
        def act(self, obs):
            mean, std, _ = self.forward(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob
        
        def evaluate(self, obs, action):
            mean, std, value = self.forward(obs)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return log_prob, value.squeeze(-1), entropy
    
    # Flatten observation for MLP
    def flatten_obs(obs):
        parts = [
            obs["position"],
            obs["velocity"],
            obs["pen_state"],
            obs["path_progress"],
            obs["steps_remaining"] / 500.0,
            obs["current_target"],
        ]
        return np.concatenate(parts, dtype=np.float32)
    
    obs_dim = len(flatten_obs(env.reset()[0]))
    act_dim = env.action_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    
    # PPO hyperparams
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    epochs = 4
    batch_size = 64
    
    # Storage
    obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = [], [], [], [], [], []
    
    print(f"Training on {device} for {total_timesteps} steps...")
    
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(total_timesteps):
        obs_flat = flatten_obs(obs)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, log_prob = net.act(obs_tensor)
            _, _, value = net.forward(obs_tensor)
        
        action_np = action.cpu().numpy().squeeze()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        # Store
        obs_buf.append(obs_flat)
        act_buf.append(action_np)
        rew_buf.append(reward)
        logp_buf.append(log_prob.item())
        val_buf.append(value.item())
        done_buf.append(done)
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            # PPO update
            if len(obs_buf) >= batch_size:
                update(net, optimizer, obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf, gamma, gae_lambda, clip_eps, epochs, batch_size, device)
                obs_buf.clear()
                act_buf.clear()
                rew_buf.clear()
                logp_buf.clear()
                val_buf.clear()
                done_buf.clear()
            
            print(f"Episode done | Steps: {step} | Reward: {episode_reward:.1f} | Progress: {info.get('path_progress', 0):.1%}")
            obs, _ = env.reset()
            episode_reward = 0
    
    # Save
    torch.save(net.state_dict(), "custom_ppo_model.pt")
    print("Model saved to custom_ppo_model.pt")


def update(net, optimizer, obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf, gamma, gae_lambda, clip_eps, epochs, batch_size, device):
    import torch
    
    obs_tensor = torch.FloatTensor(np.array(obs_buf)).to(device)
    act_tensor = torch.FloatTensor(np.array(act_buf)).to(device)
    old_logp = torch.FloatTensor(logp_buf).to(device)
    old_val = torch.FloatTensor(val_buf).to(device)
    
    # Compute returns and advantages (GAE)
    returns = []
    advantages = []
    gae = 0
    next_val = 0
    
    for r, v, d in zip(reversed(rew_buf), reversed(val_buf), reversed(done_buf)):
        delta = r + gamma * next_val * (1 - d) - v
        gae = delta + gamma * gae_lambda * (1 - d) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + v)
        next_val = v
    
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    dataset_size = len(obs_buf)
    indices = np.arange(dataset_size)
    
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            idx = indices[start:start+batch_size]
            
            mb_obs = obs_tensor[idx]
            mb_act = act_tensor[idx]
            mb_old_logp = old_logp[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]
            
            new_logp, new_val, entropy = net.evaluate(mb_obs, mb_act)
            
            ratio = (new_logp - mb_old_logp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((new_val - mb_ret) ** 2).mean()
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()


def evaluate_model(model_path: str, env, n_episodes: int = 5):
    """Evaluate a trained model."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("SB3 not available for evaluation")
        return
    
    model = PPO.load(model_path)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            env.render()
        print(f"Episode {ep+1}: Reward = {ep_reward:.1f}, Progress = {info.get('path_progress', 0):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Writing Robot RL Training")
    parser.add_argument("--env", choices=["sim", "real"], default="sim", help="Environment type")
    parser.add_argument("--algo", choices=["ppo", "custom"], default="ppo", help="Algorithm")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default=None, help="Model to load/continue")
    parser.add_argument("--port", type=str, default=None, help="Serial port (for real env)")
    parser.add_argument("--eval", action="store_true", help="Evaluate mode")
    parser.add_argument("--shape", type=str, default="square", help="Trajectory shape")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()
    
    # Trajectory
    shapes = generate_basic_shapes()
    if args.shape in shapes:
        trajectory = shapes[args.shape]
    else:
        trajectory = shapes["square"]
    
    # Environment kwargs
    env_kwargs = {
        "trajectory": trajectory,
        "render_mode": None if args.no_render else "human",
    }
    if args.port:
        env_kwargs["port"] = args.port
    
    env = make_env(args.env, **env_kwargs)
    
    if args.eval:
        if not args.model:
            print("Need --model for evaluation")
            return
        evaluate_model(args.model, env)
        return
    
    # Load config
    config = None
    if args.config:
        config = load_config(args.config)
    
    if args.algo == "ppo":
        train_ppo(env, args.timesteps, config, args.model)
    else:
        train_custom(env, args.timesteps, config or {})
    
    env.close()


if __name__ == "__main__":
    main()