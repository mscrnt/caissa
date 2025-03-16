# path: expert_env.py

import os
import numpy as np
import chess
import logging
import random
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from stockfish import Stockfish
import math  # ✅ Import math for reward normalization

# ✅ Ensure directories exist
LOGS_DIR = "./logs"
TRAJECTORIES_DIR = "/mnt/trajectories"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)

# ✅ Configure logging (Rotating Log File)
log_file = os.path.join(LOGS_DIR, "expert_env.log")
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - [Stockfish Self-Play] %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class ChessStockfishExpert:
    """Class to generate and save expert trajectories using dynamically adjusted Stockfish settings."""

    def __init__(self, stockfish_path="/usr/local/bin/stockfish", max_level=20, hash_size=12288, threads=10):
        """Initialize Stockfish experts with dynamic skill level and depth adjustment."""
        self.stockfish_path = stockfish_path
        self.hash_size = hash_size
        self.threads = threads
        self.current_color = random.choice([chess.WHITE, chess.BLACK])

    def dynamic_depth(self):
        rand = random.random()
        if rand < 0.10:  # 10% chance they have the same depth
            shared_depth = random.choice(range(10, 21))
            return shared_depth, shared_depth
        elif rand < 0.90:  # 80% chance main engine has a higher depth
            main_depth = random.choice(range(15, 21))
            opponent_depth = random.choice(range(10, main_depth + 1))
            return main_depth, opponent_depth
        else:
            return random.choice(range(10, 21)), random.choice(range(10, 21))

    def dynamic_skill_level(self):
        """Dynamically selects a skill level based on weighted probability."""
        main_levels = [15, 16, 17, 18, 19, 20]
        main_weights = [5, 5, 10, 15, 20, 45]
        main_skill = random.choices(main_levels, main_weights)[0]

        opponent_levels = list(range(1, 21))
        opponent_weights = [2] * 10 + [8] * 10
        opponent_skill = random.choices(opponent_levels, opponent_weights)[0]

        return main_skill, opponent_skill

    def get_stockfish_eval(self, stockfish_engine):
        """Gets the centipawn evaluation for the current board state."""
        eval_data = stockfish_engine.get_evaluation()
        if eval_data["type"] == "cp":
            return eval_data["value"] / 100
        elif eval_data["type"] == "mate":
            return 10 if eval_data["value"] > 0 else -10
        return 0

    def evaluate_turn(self, board, main_color, eval_before, eval_after, stockfish_engine):
        """Evaluates move-based and final rewards for the Main player."""
        delta_eval = eval_after - eval_before
        reward = 2 / (1 + math.exp(-delta_eval)) - 1  # Sigmoid normalization

        if delta_eval < -2.0:
            reward -= 0.5  

        if board.is_checkmate():
            reward = 5.0 if board.turn != main_color else -5.0
            return reward, reward

        if board.is_stalemate():
            reward = -0.5  
        elif board.is_insufficient_material():
            reward = 0  
        elif board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            reward = -0.5  

        win, draw, loss = stockfish_engine.get_wdl_stats()
        total = win + draw + loss
        if total > 0:
            win_prob = win / total
            loss_prob = loss / total
            reward += (win_prob - loss_prob) * 2

        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner == main_color:
                final_reward = 1 + min(1, abs(eval_after) / 10)
            elif outcome.winner is None:
                final_reward = 0
            else:
                final_reward = -1 - min(1, abs(eval_after) / 10)
            return reward, final_reward

        return reward, None

    def record_game(self, game_id, env_id):
        """Runs a Stockfish self-play game with dynamically assigned skill levels and depths."""
        board = chess.Board()
        main_skill, opponent_skill = self.dynamic_skill_level()
        main_depth, opponent_depth = self.dynamic_depth()
        self.current_color = random.choice([chess.WHITE, chess.BLACK])

        log_message = (f"♟️ [Env {env_id}] Game {game_id}: "
                       f"Main (Level {main_skill}, Depth {main_depth}) vs. "
                       f"Opponent (Level {opponent_skill}, Depth {opponent_depth}) | "
                       f"Main plays {'White' if self.current_color == chess.WHITE else 'Black'}")
        logger.info(log_message)
        tqdm.write(log_message)

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

        while not board.is_game_over():
            current_fen = board.fen()
            is_main_move = board.turn == self.current_color
            current_engine = stockfish_main if is_main_move else stockfish_opponent
            current_engine.set_fen_position(current_fen)

            eval_before = self.get_stockfish_eval(current_engine)
            best_move = current_engine.get_best_move()

            if not best_move:
                logger.warning(f"🚨 [Env {env_id}] No move found! FEN: {current_fen}")
                break

            board.push(chess.Move.from_uci(best_move))
            eval_after = self.get_stockfish_eval(current_engine)
            last_eval = eval_after  # ✅ Update last_eval correctly

            reward, final_reward = self.evaluate_turn(board, self.current_color, eval_before, eval_after, current_engine)

            trajectory.append((current_fen, best_move, board.fen(), reward, board.is_game_over(), board.turn))

            if final_reward is not None:
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], final_reward, trajectory[-1][4], trajectory[-1][5])
                break  # ✅ Stop recording if the game ends

            logger.info(f"📌 [Env {env_id}] Move {len(trajectory)} ({'Main' if is_main_move else 'Opponent'}): {best_move} | Reward: {reward}")

        # ✅ Ensure final reward is correctly assigned
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner == self.current_color:
                final_reward = 1 + min(1, abs(last_eval) / 10)
            elif outcome.winner is None:
                final_reward = 0
            else:
                final_reward = -1 - min(1, abs(last_eval) / 10)

            trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], final_reward, trajectory[-1][4], trajectory[-1][5])

        logger.info(f"🏁 [Env {env_id}] Game {game_id} Over! Result: {board.result()} | Final Reward: {final_reward}")
        return trajectory


    def save_expert_data(self, num_games=1000, env_id=0):
        with tqdm(total=num_games, desc=f"[Env {env_id}] Generating Games", unit="game") as pbar:
            for game_id in range(1, num_games + 1):
                game_filename = os.path.join(TRAJECTORIES_DIR, f"env_{env_id}_game_{game_id:06d}.npz")

                game_data = self.record_game(game_id, env_id)

                expert_data = {
                    "obs": np.array([obs for obs, _, _, _, _, _ in game_data]),
                    "actions": np.array([action for _, action, _, _, _, _ in game_data]),
                    "next_obs": np.array([next_obs for _, _, next_obs, _, _, _ in game_data]),
                    "rewards": np.array([reward for _, _, _, reward, _, _ in game_data]),
                    "dones": np.array([done for _, _, _, _, done, _ in game_data]),
                    "colors": np.array(["white" if color == chess.WHITE else "black" for _, _, _, _, _, color in game_data]),
                }

                np.savez_compressed(game_filename, **expert_data)
                logger.info(f"✅ [Env {env_id}] Saved: {game_filename}")

                self.current_color = not self.current_color
                pbar.update(1)

if __name__ == "__main__":
    chess_expert = ChessStockfishExpert()
    chess_expert.save_expert_data(num_games=500)
