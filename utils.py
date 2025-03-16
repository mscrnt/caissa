################################################################################
#
# utils.py
# Utility functions for Caissa.
# Handles logging, Stockfish settings, evaluation, and game mechanics.
#
# © 2025, Kenneth Blossom
################################################################################

import random
import math
import logging
import os
import glob
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import chess

logger = logging.getLogger(__name__)  # ✅ Ensure logging is available

def flip_fen(fen):
    """
    Flips a FEN string to swap white and black perspectives.

    :param fen: (str) The FEN string to flip.
    :return: (str) The flipped FEN.
    """
    board = chess.Board(fen)

    # Flip the board perspective
    board = board.mirror()

    # Swap turn indicator
    parts = fen.split(" ")
    parts[1] = "w" if parts[1] == "b" else "b"

    # Update the castling rights (flip king and queen side)
    castling = parts[2]
    flipped_castling = castling.translate(str.maketrans("KQkq", "kqKQ"))
    parts[2] = flipped_castling

    # Keep everything else the same but with a flipped board
    return board.fen()

def uci_to_action_index(uci_move, env):
    """
    Converts a UCI move (e.g., 'e2e4') to an action index used by gym-chess.

    :param uci_move: (str) The move in UCI notation.
    :param env: (gym.Env) The gym chess environment.
    :return: (int) The corresponding action index.
    """
    move_list = list(env.legal_moves)
    if uci_move in move_list:
        return move_list.index(uci_move)
    return None  # Invalid move

def action_index_to_uci(action_index, env):
    """
    Converts an action index to a UCI move string.

    :param action_index: (int) The action index from the agent.
    :param env: (gym.Env) The gym chess environment.
    :return: (str) The corresponding UCI move.
    """
    move_list = list(env.legal_moves)
    if 0 <= action_index < len(move_list):
        return move_list[action_index]
    return None  # Invalid action

# Linear scheduler for RL agent parameters
def linear_schedule(initial_value, final_value=0.0):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0), "linear_schedule work only with positive decreasing values"

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return final_value + progress * (initial_value - final_value)

    return func

# AutoSave Callback
class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :filename_prefix: (str) Filename prefix
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, num_envs: int, save_path: str, filename_prefix: str="", verbose: int=1):
        super(AutoSave, self).__init__(verbose)
        self.check_freq = int(check_freq / num_envs)
        self.num_envs = num_envs
        self.save_path_base = Path(save_path)
        self.filename = filename_prefix + "autosave_"

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print("Saving latest model to {}".format(self.save_path_base))
            # Save the agent
            self.model.save(self.save_path_base / (self.filename + str(self.n_calls * self.num_envs)))

        return True

def dynamic_skill_level():
    """Dynamically selects a skill level based on weighted probability."""
    main_levels = [15, 16, 17, 18, 19, 20]
    main_weights = [5, 5, 10, 15, 20, 45]
    main_skill = random.choices(main_levels, main_weights)[0]

    opponent_levels = list(range(1, 21))
    opponent_weights = [2] * 10 + [8] * 10
    opponent_skill = random.choices(opponent_levels, opponent_weights)[0]

    return main_skill, opponent_skill


def dynamic_depth():
    """Assigns different depth levels for Stockfish players based on a probability distribution."""
    rand = random.random()
    if rand < 0.10:  # 10% chance both engines have the same depth
        shared_depth = random.choice(range(10, 21))
        return shared_depth, shared_depth
    elif rand < 0.90:  # 80% chance main engine has a higher depth
        main_depth = random.choice(range(15, 21))
        opponent_depth = random.choice(range(10, main_depth + 1))
        return main_depth, opponent_depth
    else:
        return random.choice(range(10, 21)), random.choice(range(10, 21))


def get_stockfish_eval(stockfish_engine):
    """Gets the centipawn evaluation for the current board state."""
    eval_data = stockfish_engine.get_evaluation()
    if eval_data["type"] == "cp":
        return eval_data["value"] / 100
    elif eval_data["type"] == "mate":
        return 10 if eval_data["value"] > 0 else -10
    return 0


def evaluate_turn(board, main_color, eval_before, eval_after, stockfish_engine):
    """Evaluates move-based and final rewards for the Main player."""
    delta_eval = eval_after - eval_before
    reward = 2 / (1 + math.exp(-delta_eval)) - 1  # Sigmoid normalization

    if delta_eval < -2.0:
        reward -= 0.5  # ✅ Extra penalty for large drop in evaluation

    # ✅ Debugging output for evaluation tracking
    logger.info(f"📊 Evaluating move: ΔEval={delta_eval:.2f}, Move Reward={reward:.2f}")

    # ✅ Check game outcome
    outcome = board.outcome()
    if outcome is not None:
        winner = outcome.winner  # ✅ Get actual winner of the game

        if winner == main_color:
            final_reward = 5.0  # ✅ Strong reward for winning
        elif winner is None:  # Draw
            final_reward = -0.5  # ✅ Small penalty for draws
        else:
            final_reward = -5.0  # ✅ Strong penalty for losing

        logger.info(f"🏁 Final game result detected! Winner: {'Main' if winner == main_color else 'Opponent' if winner is not None else 'Draw'} | Final Reward: {final_reward:.2f}")

        return reward, final_reward  # ✅ Ensure last move gets correct reward

    # ✅ Handle draw-related scenarios
    if board.is_stalemate():
        reward = -0.5  # ✅ Stalemate = negative reward
    elif board.is_insufficient_material():
        reward = 0  # ✅ Insufficient material = neutral reward
    elif board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
        reward = -0.5  # ✅ Repetitions = negative reward

    # ✅ Adjust based on Stockfish's WDL probabilities
    wdl_stats = stockfish_engine.get_wdl_stats()
    if wdl_stats and len(wdl_stats) == 3:
        win, draw, loss = wdl_stats
        total = win + draw + loss
        if total > 0:
            win_prob = win / total
            loss_prob = loss / total
            reward += (win_prob - loss_prob) * 2

    return reward, None  # ✅ Return move reward, but no final reward yet


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