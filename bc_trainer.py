#!/usr/bin/env python
################################################################################
#
# bc_trainer.py
# Pretrains a chess RL agent using Behavior Cloning (BC) with Stable Baselines3 (SB3).
#
# © 2025, Kenneth Blossom
################################################################################

import os
import torch
import logging
import numpy as np
import gymnasium as gym
import glob
from logging.handlers import RotatingFileHandler
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from imitation.algorithms.bc import BC
from imitation.data import types
from utils import (
    flip_fen, uci_to_action_index, linear_schedule, AutoSave, load_npz_trajectories
)  # ✅ Import conversion & scheduling utilities

# ✅ Directories
LOG_DIR = "./logs"
TRAJECTORIES_DIR_NPZ = "./output/trajectories"
TRAJECTORIES_DIR_PT = "./output/trajectories_gpu"
MODEL_SAVE_PATH = "./models/ppo_bc_chess.zip"

# ✅ Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRAJECTORIES_DIR_PT, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ✅ Setup logging
log_file = os.path.join(LOG_DIR, "bc_trainer.log")
log_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.addHandler(console_handler)

# ✅ Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Training on: {device}")

# ✅ Load Expert Trajectories
def load_trajectories(npz_dir, pt_dir, device):
    """Loads expert trajectories from either .npz (CPU) or .pt (GPU) files dynamically."""
    if device.type == "cuda" and any(f.endswith(".pt") for f in os.listdir(pt_dir)):
        logger.info("🔄 Loading expert trajectories from .pt files (GPU mode)")
        files = glob.glob(os.path.join(pt_dir, "*.pt"))
        load_func = torch.load
    else:
        logger.info("🖥️ Loading expert trajectories from .npz files (CPU mode)")
        files = glob.glob(os.path.join(npz_dir, "*.npz"))
        load_func = np.load

    if not files:
        logger.error("🚨 No trajectory files found!")
        return None

    obs_list, actions_list = [], []
    for file in files:
        try:
            data = load_func(file)
            obs_list.append(torch.tensor(data["obs"], dtype=torch.float32) if device.type == "cuda" else data["obs"])
            actions_list.append(torch.tensor(data["actions"], dtype=torch.int64) if device.type == "cuda" else data["actions"])
        except Exception as e:
            logger.warning(f"⚠️ Skipping {file} due to error: {e}")

    obs_array = torch.cat(obs_list, dim=0) if device.type == "cuda" else np.concatenate(obs_list, axis=0)
    actions_array = torch.cat(actions_list, dim=0) if device.type == "cuda" else np.concatenate(actions_list, axis=0)

    logger.info(f"📊 Loaded {obs_array.shape[0]} expert moves from {len(files)} files")
    return obs_array, actions_array

# ✅ Create Gym-Chess Environment
def make_chess_env():
    """Creates a vectorized gym-chess environment."""
    env = make_vec_env("Chess-v0", n_envs=1, seed=42)
    return env

# ✅ Train BC Model
def train_bc_model(epochs=10, batch_size=512, lr=1e-4, save_freq=50000):
    """Trains PPO model using Behavior Cloning (BC) from expert data."""
    data = load_trajectories(TRAJECTORIES_DIR_NPZ, TRAJECTORIES_DIR_PT, device)

    if data is None:
        logger.error("❌ No expert trajectories found! Exiting training.")
        return

    obs_tensor, actions_tensor = data  # ✅ Safe unpacking

    env = make_chess_env()

    # ✅ Use linear schedule for learning rate
    lr_schedule = linear_schedule(lr, final_value=1e-6)

    model = PPO("MlpPolicy", env, verbose=1, device=device, learning_rate=lr_schedule)

    # ✅ Auto-save callback for regular model checkpoints
    autosave_callback = AutoSave(
        check_freq=save_freq, num_envs=1, save_path="./models/", filename_prefix="ppo_bc_"
    )

    dataset = types.Transitions(
        obs=obs_tensor.cpu().numpy(),
        acts=actions_tensor.cpu().numpy(),
        infos=None,
        next_obs=None,
        dones=None,
        rews=None,
    )

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=dataset,
        policy=model.policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        device=device,
    )

    logger.info(f"🎯 Starting Behavior Cloning training for {epochs} epochs...")
    bc_trainer.train(n_epochs=epochs)

    model.save(MODEL_SAVE_PATH)
    logger.info(f"✅ PPO model pre-trained with BC saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc_model()