import os
import time
import logging
import multiprocessing as mp
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from chess_environment import ChessEnv

def make_sb3_env(num_envs=16, seed=42, use_subprocess=True, log_dir_base="/tmp/DIAMBRALog/", start_index=0):
    """
    Create a wrapped, monitored VecEnv for ChessRL.
    
    :param num_envs: (int) Number of environments to create
    :param seed: (int) Random seed for reproducibility
    :param use_subprocess: (bool) Use SubprocVecEnv for multiprocessing
    :param log_dir_base: (str) Base directory for logs
    :param start_index: (int) Starting index for environment numbering
    :return: (VecEnv) The created Stable Baselines3 environment
    """

    # Ensure log directory exists
    os.makedirs(log_dir_base, exist_ok=True)

    def _make_chess_env(rank, seed):
        """Helper function to create a single ChessEnv."""
        env_seed = int(time.time()) if seed is None else seed + rank

        def _init():
            env = ChessEnv(stockfish_path="/usr/games/stockfish", stockfish_skill=20, env_id=rank)
            env.reset(seed=env_seed)

            # Create per-env log directory
            log_dir = os.path.join(log_dir_base, f"env_{rank}")
            os.makedirs(log_dir, exist_ok=True)

            return Monitor(env, log_dir)
        
        set_random_seed(env_seed)
        return _init

    # Use DummyVecEnv for single env, otherwise use SubprocVecEnv
    if num_envs == 1 or not use_subprocess:
        env = DummyVecEnv([_make_chess_env(i + start_index, seed) for i in range(num_envs)])
    else:
        env = SubprocVecEnv([_make_chess_env(i + start_index, seed) for i in range(num_envs)], start_method="forkserver")

    return env, num_envs
