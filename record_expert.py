#!/usr/bin/env python
################################################################################
#
# chess_expert.py
# Main script for running Caissa with parallel environments.
#
# © 2025, Kenneth Blossom
################################################################################

import os
import logging
import multiprocessing as mp
import numpy as np
import datetime
import csv
import time
import chess
import chess.syzygy
import random
from tqdm import tqdm
from stockfish import Stockfish
from logging.handlers import RotatingFileHandler
from utils import (
    dynamic_skill_level,
    dynamic_depth,
    get_stockfish_eval,
    evaluate_turn,
)

# ✅ Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# ✅ Setup log rotation
log_file = './logs/caissa_expert.log'
log_handler = RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10 MB per log file
    backupCount=20  # Keep up to 20 backups
)

# ✅ Ensure logging works nicely with tqdm
class TqdmLoggingHandler(logging.Handler):
    """Custom handler to print logs with tqdm while also writing to the log file."""
    
    def __init__(self, file_handler, level=logging.NOTSET):
        super().__init__(level)
        self.file_handler = file_handler  # ✅ Forward logs to file

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # ✅ Keeps progress bar at the bottom
            self.file_handler.emit(record)  # ✅ Write log to file
        except Exception:
            self.handleError(record)

# ✅ Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ✅ Check if handlers are already added (avoid duplicate logs)
if not logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler.setFormatter(log_formatter)
    tqdm_handler = TqdmLoggingHandler(log_handler)
    tqdm_handler.setFormatter(log_formatter)
    
    # Option 2: Remove the separate file handler from the logger.
    # Instead of adding both, we only add the tqdm_handler which handles both console and file logging.
    # logger.addHandler(log_handler)   # Remove or comment this line.
    logger.addHandler(tqdm_handler)

# ✅ Generate a unique identifier for this run
TRAJECTORIES_DIR = "./output/trajectories"
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)

# ✅ Multiprocessing environment settings
num_envs = 1  # ✅ Adjust the number of parallel environments
num_games_per_env = 100000 // num_envs  # ✅ Distribute games across environments
total_games = num_envs * num_games_per_env  # ✅ Total number of games


class ChessStockfishExpert:
    """Class to generate and save expert trajectories using dynamically adjusted Stockfish settings."""

    def __init__(self, stockfish_path="/usr/local/bin/stockfish", syzygy_path="./syzygy"):
        self.stockfish_path = stockfish_path
        self.current_color = chess.WHITE  # ✅ Alternate each game

        # ✅ Load Syzygy tablebases from multiple directories
        self.tablebase = chess.syzygy.Tablebase()
        wdl_count, dtz_count = 0, 0

        try:
            # ✅ Load 6-piece Syzygy tablebases
            wdl_count += self.tablebase.add_directory(os.path.join(syzygy_path, "wdl"))
            dtz_count += self.tablebase.add_directory(os.path.join(syzygy_path, "dtz"))

            # ✅ Load 3-4-5-piece Syzygy tablebases
            wdl_count += self.tablebase.add_directory(os.path.join(syzygy_path, "345"))
            dtz_count += self.tablebase.add_directory(os.path.join(syzygy_path, "345"))

            logger.info(f"✅ Loaded {wdl_count} WDL tables, {dtz_count} DTZ tables from {syzygy_path}")
        except Exception as e:
            logger.error(f"🚨 Failed to load Syzygy tablebases: {e}")
            self.tablebase = None  # ✅ Fail gracefully

    def probe_tablebase(self, board):
        """Probes Syzygy tablebase only if the position has 6 or fewer total pieces and no castling rights."""
        if self.tablebase is None:
            return None  # ✅ No tablebase loaded

        if board.is_game_over():
            return None  # ✅ No need to probe

        # ✅ Count all pieces on the board (White + Black)
        num_pieces = len(board.piece_map())

        # ✅ Syzygy supports 6 pieces or fewer (including kings)
        if num_pieces > 6:
            return None  # ✅ Skip probing if too many pieces

        # ✅ Skip positions with castling rights (Syzygy does not support them)
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return None

        try:
            wdl = self.tablebase.probe_wdl(board)
            dtz = self.tablebase.probe_dtz(board)
            logger.info(f"📖 Syzygy probe: WDL={wdl}, DTZ={dtz} for FEN: {board.fen()}")
            return wdl, dtz
        except chess.syzygy.MissingTableError:
            logger.warning(f"🚨 Syzygy tablebase missing for FEN: {board.fen()}")
            return None  # ✅ Position not in tablebase
        except KeyError:
            logger.warning(f"🚨 Syzygy probe failed due to missing tablebase file for FEN: {board.fen()}")
            return None
        except Exception as e:
            logger.error(f"🚨 Unexpected error during Syzygy probing: {e}")
            return None

    def record_game(self, game_id, env_id):
        """Runs a Stockfish self-play game with dynamically assigned skill levels and depths,
        and returns both the game trajectory and a dictionary of game statistics.
        """
        board = chess.Board()
        # Save the initial main player color since it will be toggled later.
        initial_main_color = self.current_color
        main_skill, opponent_skill = dynamic_skill_level()
        main_depth, opponent_depth = dynamic_depth()

        logger.info(
            f"♟️ [Env {env_id}] Game {game_id}: "
            f"Main (Level {main_skill}, Depth {main_depth}) vs. "
            f"Opponent (Level {opponent_skill}, Depth {opponent_depth}) | "
            f"Main plays {'White' if initial_main_color == chess.WHITE else 'Black'}"
        )

        # Create Stockfish instances.
        stockfish_main = Stockfish(self.stockfish_path, parameters={
            "Skill Level": main_skill, "Hash": 16384, "Threads": 8, "MultiPV": 1, "Contempt": 0
        })
        stockfish_main.set_depth(main_depth)

        stockfish_opponent = Stockfish(self.stockfish_path, parameters={
            "Skill Level": opponent_skill, "Hash": 8192, "Threads": 2, "MultiPV": 1, "Contempt": 0
        })
        stockfish_opponent.set_depth(opponent_depth)

        trajectory = []
        last_eval = 0
        total_reward = 0
        move_count = 0

        while not board.is_game_over():
            move_count += 1
            current_fen = board.fen()
            # Use the initial main color to decide whose move is considered "main"
            is_main_move = board.turn == initial_main_color
            current_engine = stockfish_main if is_main_move else stockfish_opponent
            current_engine.set_fen_position(current_fen)
            eval_before = get_stockfish_eval(current_engine)

            # Syzygy tablebase probing (if possible)
            tb_result = self.probe_tablebase(board)
            if tb_result:
                wdl, dtz = tb_result
                logger.info(f"📖 [Env {env_id}] Syzygy probe: WDL={wdl}, DTZ={dtz} for FEN: {current_fen}")

            # Retry get_best_move() if it fails
            best_move = None
            for _ in range(3):
                best_move = current_engine.get_best_move()
                if best_move:
                    break
                time.sleep(0.1)

            if not best_move:
                logger.warning(f"🚨 [Env {env_id}] No move found! FEN: {current_fen}")
                break

            board.push(chess.Move.from_uci(best_move))
            eval_after = get_stockfish_eval(current_engine)
            last_eval = eval_after

            reward, final_reward = evaluate_turn(board, initial_main_color, eval_before, eval_after, current_engine)
            total_reward += reward

            trajectory.append((current_fen, best_move, board.fen(), reward, board.is_game_over(), board.turn))

            if final_reward is not None:
                # Overwrite the last move's reward with the final reward.
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], final_reward, trajectory[-1][4], trajectory[-1][5])
                total_reward += final_reward
                break

            player_role = "Main" if is_main_move else "Opponent"
            player_skill = main_skill if is_main_move else opponent_skill
            player_depth = main_depth if is_main_move else opponent_depth

            logger.info(
                f"📌 [Env {env_id}] Move {move_count} ({player_role}, Level {player_skill}, Depth {player_depth}): "
                f"{best_move} | Move Reward: {reward:.3f} | Total Score: {total_reward:.3f}"
            )

        # Ensure final reward is properly assigned if not already done.
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner == initial_main_color:
                final_reward = 1 + min(1, abs(last_eval) / 10)
            elif outcome.winner is None:
                final_reward = 0
            else:
                final_reward = -1 - min(1, abs(last_eval) / 10)

            trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], final_reward, trajectory[-1][4], trajectory[-1][5])
            total_reward += final_reward

        logger.info(
            f"🏁 [Env {env_id}] Game {game_id} Over! Result: {board.result()} | "
            f"Final Move Reward: {final_reward:.3f} | Total Score: {total_reward:.3f} | Moves Played: {move_count}"
        )

        # Clean up Stockfish instances.
        del stockfish_main
        del stockfish_opponent

        # Alternate colors for the next game.
        self.current_color = not self.current_color

        game_stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file": None,  # Will be filled in after saving the file.
            "result": board.result(),
            "total_reward": total_reward,
            "main_skill": main_skill,
            "main_depth": main_depth,
            "opponent_skill": opponent_skill,
            "opponent_depth": opponent_depth,
            "move_count": move_count,
            "main_color": "White" if initial_main_color == chess.WHITE else "Black"
        }

        return trajectory, game_stats

# Modified run_expert_env now writes to a shared CSV file.
def run_expert_env(env_id, progress_queue, csv_lock, stats_filename):
    base_dir = TRAJECTORIES_DIR
    expert = ChessStockfishExpert()
    logger.info(f"🎮 Starting Expert Env {env_id}, saving in {base_dir}")

    for i in range(1, num_games_per_env + 1):
        unique_suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        game_filename = os.path.join(base_dir, f"env_{env_id}_game_{i:06d}_{unique_suffix}.npz")
        result = expert.record_game(i, env_id)
        if not result:
            continue
        game_data, game_stats = result

        expert_data = {
            "obs": np.array([obs for obs, _, _, _, _, _ in game_data]),
            "actions": np.array([action for _, action, _, _, _, _ in game_data]),
            "next_obs": np.array([next_obs for _, _, next_obs, _, _, _ in game_data]),
            "rewards": np.array([reward for _, _, _, reward, _, _ in game_data]),
            "dones": np.array([done for _, _, _, _, done, _ in game_data]),
            "colors": np.array(["white" if color == chess.WHITE else "black" for _, _, _, _, _, color in game_data]),
        }
        np.savez_compressed(game_filename, **expert_data)
        game_stats["file"] = game_filename

        # Use a lock to safely append to the shared CSV file.
        with csv_lock:
            with open(stats_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "timestamp", "file", "result", "total_reward", "main_skill", "main_depth",
                    "opponent_skill", "opponent_depth", "move_count", "main_color"
                ])
                writer.writerow(game_stats)

        progress_queue.put(1)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Define a shared CSV file path and create it with header before starting processes.
    stats_filename = os.path.join(TRAJECTORIES_DIR, "game_stats.csv")
    header = ["timestamp", "file", "result", "total_reward", "main_skill", "main_depth",
              "opponent_skill", "opponent_depth", "move_count", "main_color"]
    if not os.path.exists(stats_filename):
        with open(stats_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # Create a multiprocessing lock to synchronize CSV writes.
    csv_lock = mp.Lock()

    progress_queue = mp.Queue()
    processes = [mp.Process(target=run_expert_env, args=(env_id, progress_queue, csv_lock, stats_filename))
                 for env_id in range(num_envs)]

    for p in processes:
        p.start()

    with tqdm(total=total_games, desc="Generating Chess Trajectories", unit="game") as pbar:
        for _ in range(total_games):
            progress_queue.get()
            pbar.update(1)

    for p in processes:
        p.join()

    logger.info(f"✅ All expert trajectory environments completed! Saved in {TRAJECTORIES_DIR}")
