import gymnasium as gym
import numpy as np
import chess
import chess.engine
from gymnasium import spaces
from stockfish import Stockfish
import logging
import time
import random

# ✅ Set up cleaner logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Env %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ChessEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}  # ✅ Define metadata for Gymnasium compliance

    def __init__(self, stockfish_path="/usr/games/stockfish", stockfish_skill=2, env_id=0):
        super(ChessEnv, self).__init__()

        self.stockfish_path = stockfish_path
        self.stockfish_skill = stockfish_skill
        self.env_id = env_id
        self.move_count = 0
        self.episode_rewards = 0
        self.board = chess.Board()
        self.move_history = []
        self.game_start_time = time.time()  # ✅ Track game duration
        self.invalid_move_attempts = set()  # ✅ Track unique invalid move attempts

        # ✅ Define action and observation spaces
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Dict(
            {
                "board_state": spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=np.float32),
                "wdl_stats": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32),
                "move_history": spaces.MultiDiscrete([4672] * 200),
                "valid_moves_mask": spaces.Box(low=0, high=1, shape=(4672,), dtype=np.float32),  # ✅ Add this
            }
        )


        # ✅ Initialize Stockfish
        self.stockfish = Stockfish(self.stockfish_path, parameters={"Skill Level": self.stockfish_skill})
        self.stockfish.set_fen_position(self.board.fen())

        self.supports_wdl = self.stockfish.does_current_engine_version_have_wdl_option()


    def reset(self, seed=None, options=None):
        """Resets the environment and assigns the agent a random color."""
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)  # ✅ Ensure reproducibility

        self.board.reset()
        self.move_count = 0
        self.move_history = []
        self.episode_rewards = 0
        self.game_start_time = time.time()  # ✅ Restart game timer
        self.invalid_attempts = 0  # ✅ Reset invalid move counter

        # ✅ Randomly assign the agent to play as White or Black
        self.agent_color = random.choice([chess.WHITE, chess.BLACK])

        logging.info(f"[Env {self.env_id}] ♻️ Environment Reset. Agent is playing {'White' if self.agent_color else 'Black'}")

        # ✅ If the agent is Black, let Stockfish make the first move
        if self.agent_color == chess.BLACK:
            stockfish_move = self.stockfish.get_best_move()
            if stockfish_move:
                stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                if stockfish_move_obj in self.board.legal_moves:
                    self.board.push(stockfish_move_obj)
                    self.move_history.append(stockfish_move_obj)
                    self.stockfish.set_fen_position(self.board.fen())
                    logging.info(f"[Env {self.env_id}] 🤖 Stockfish Move: {stockfish_move} (First move as White)")

        return self._get_obs(), self._get_info()


    def step(self, action):
        """Executes a move with soft action masking, allowing learning of invalid moves."""
        try:
            # ✅ Ensure it's the agent's turn to move
            if self.board.turn != self.agent_color:
                #logging.warning(f"[Env {self.env_id}] 🚫 Agent tried to move out of turn!")
                return self._get_obs(), -10, False, False, self._get_info()

            legal_moves = list(self.board.legal_moves)
            legal_move_indices = [self._move_to_index(m) for m in legal_moves]

            # ✅ If the selected move is invalid, apply a penalty but allow learning
            if action not in legal_move_indices:
                self.invalid_attempts += 1
                penalty = -10 * self.invalid_attempts  # Increase penalty for repeated failures

                # ✅ Truncate if conditions are met
                if self._check_truncation():
                    return self._get_obs(), penalty, False, True, self._get_info()

                return self._get_obs(), penalty, False, False, self._get_info()

            # ✅ Execute the valid move
            move = legal_moves[legal_move_indices.index(action)]
            self.board.push(move)
            self.move_count += 1
            self.move_history.append(move)

            logging.info(
                f"[Env {self.env_id}] ✅ Move #{self.move_count} - Agent Move: {move.uci()} "
                f"(Invalid Attempts Before Success: {self.invalid_attempts})"
            )

            # ✅ Reset invalid attempts after a valid move
            self.invalid_attempts = 0  

            # ✅ Sync Stockfish position
            self.stockfish.set_fen_position(self.board.fen())

            # ✅ Reward agent for choosing a valid move
            base_reward = self._evaluate_board()
            valid_move_bonus = 10
            reward = base_reward + valid_move_bonus  
            self.episode_rewards += reward

            # ✅ Check termination **BEFORE** Stockfish moves
            result = self._determine_result()
            terminated = result in ["WIN", "LOSS", "DRAW"]

            # ✅ Check truncation using our improved method
            truncated = self._check_truncation()

            # ✅ STOP IMMEDIATELY if the game is over (Stockfish should NOT move)
            if terminated or truncated:
                game_duration = time.time() - self.game_start_time
                logging.info(
                    f"[Env {self.env_id}] 🎯 Game Over! Result: {result} | Moves: {self.move_count} | Duration: {game_duration:.2f}s"
                )
                return self._get_obs(), reward, terminated, truncated, {"episode": {"r": self.episode_rewards, "l": self.move_count}}

            # ✅ Stockfish's move (Opponent) **ONLY IF GAME IS STILL ON**
            if self.board.turn != self.agent_color:
                stockfish_move_start = time.time()
                stockfish_move = self.stockfish.get_best_move()
                stockfish_thinking_time = time.time() - stockfish_move_start

                if stockfish_move:
                    stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                    if stockfish_move_obj in self.board.legal_moves:
                        self.board.push(stockfish_move_obj)
                        self.move_history.append(stockfish_move_obj)
                        self.stockfish.set_fen_position(self.board.fen())
                        logging.info(
                            f"[Env {self.env_id}] 🤖 Stockfish Move: {stockfish_move} (Thinking Time: {stockfish_thinking_time:.3f}s) | Move #{self.move_count}"
                        )

            return self._get_obs(), reward, terminated, truncated, self._get_info()

        except Exception as e:
            logging.error(f"[Env {self.env_id}] ❌ Error in step: {e} | Move #{self.move_count}")
            return self._get_obs(), -100, True, False, {}



    def _index_to_move(self, index):
        """Converts an action index back into a UCI move string."""
        from_square = index // 64
        to_square = index % 64
        return chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]


    def _evaluate_board(self):
        """Returns a normalized evaluation."""
        if self.board.is_checkmate():
            return 100  # ✅ Win for the agent
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0  # ✅ Draw

        eval_result = self.stockfish.get_evaluation()
        evaluation = eval_result["value"] / 100 if eval_result["type"] == "cp" else 0

        logging.info(f"[Env {self.env_id}] 📈 Stockfish Eval: {eval_result}")
        return evaluation

    def _get_obs(self):
        """Returns the current board state and a soft action mask."""
        board_matrix = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                row, col = divmod(square, 8)
                board_matrix[channel, row, col] = 1

        # ✅ Soft action mask with proper indexing and normalization
        valid_moves_mask = np.full(4672, 1e-9, dtype=np.float32)  # Low probability default
        valid_move_indices = [self._move_to_index(m) for m in self.board.legal_moves]

        for idx in valid_move_indices:
            if 0 <= idx < 4672:  # ✅ Ensure it falls within action_space bounds
                valid_moves_mask[idx] = 1.0

        valid_moves_mask /= valid_moves_mask.sum()  # ✅ Normalize

        return {
            "board_state": board_matrix,
            "wdl_stats": np.array(self.stockfish.get_wdl_stats() or [0, 0, 0], dtype=np.float32),
            "move_history": np.pad(
                np.array([m.from_square * 64 + m.to_square for m in self.move_history[-200:]], dtype=np.int32),
                (0, 200 - len(self.move_history)),
                mode="constant",
            ),
            "valid_moves_mask": valid_moves_mask,  # ✅ Soft Action Masking
        }



    def _move_to_index(self, move):
        """Converts a chess.Move object into a discrete action index."""
        return move.from_square * 64 + move.to_square


    def _get_info(self):
        """Returns additional info."""
        return {
            "centipawn_eval": self.stockfish.get_evaluation()["value"],
            "move_count": self.move_count
        }

    def _determine_result(self):
        """Determines game result for logging."""
        outcome = self.board.outcome()
        
        if outcome is None:
            return "ONGOING"  # Game is still in progress

        # Checkmate detected
        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "WIN" if outcome.winner == self.agent_color else "LOSS"
            logging.info(f"[Env {self.env_id}] 🎉 Checkmate! {winner}")
            return winner

        # Other draw conditions (stalemate, insufficient material, repetition, 50-move rule)
        if outcome.termination in {
            chess.Termination.STALEMATE,
            chess.Termination.INSUFFICIENT_MATERIAL,
            chess.Termination.FIVEFOLD_REPETITION,
            chess.Termination.SEVENTYFIVE_MOVES,
        }:
            logging.info(f"[Env {self.env_id}] 🤝 Game Drawn! ({outcome.termination.name})")
            return "DRAW"

        # If somehow an unknown termination happens (shouldn't occur)
        logging.warning(f"[Env {self.env_id}] ⚠️ Unexpected game termination: {outcome.termination}")
        return "UNKNOWN"


    def close(self):
        pass

    def _check_truncation(self):
        """Detects if the game should be truncated due to various end conditions."""
        
        # ✅ Check if the game has hit the move limit
        if self.move_count >= 60:
            logging.info(f"[Env {self.env_id}] ⏳ Move limit reached. Truncating game.")
            return True

        # ✅ Check for fivefold repetition (game automatically drawn by FIDE rules)
        if self.board.is_fivefold_repetition():
            logging.info(f"[Env {self.env_id}] 🔁 Fivefold repetition detected. Truncating game.")
            return True

        # ✅ Check for threefold repetition (can be claimed)
        if self.board.can_claim_threefold_repetition():
            logging.info(f"[Env {self.env_id}] 🔁 Threefold repetition detected. Truncating game.")
            return True

        # ✅ Check for the 75-move rule (game automatically drawn by FIDE rules)
        if self.board.is_seventyfive_moves():
            logging.info(f"[Env {self.env_id}] ⏳ 75-move rule applied. Truncating game.")
            return True

        # ✅ Check for 50-move rule (can be claimed)
        if self.board.can_claim_fifty_moves():
            logging.info(f"[Env {self.env_id}] ⏳ 50-move rule applied. Truncating game.")
            return True

        # ✅ Check for insufficient material (automatic draw if neither side can checkmate)
        if self.board.is_insufficient_material():
            logging.info(f"[Env {self.env_id}] 🤝 Draw due to insufficient material. Truncating game.")
            return True

        # ✅ Check for stalemate (opponent has no legal moves but is NOT in check)
        if self.board.is_stalemate():
            logging.info(f"[Env {self.env_id}] 🤝 Stalemate detected. Truncating game.")
            return True

        # ✅ If the agent makes too many invalid moves, truncate
        if self.invalid_attempts >= 20_000:
            logging.warning(f"[Env {self.env_id}] 🚨 Too many invalid moves! Truncating episode.")
            return True

        return False



# ✅ Register the environment
from gymnasium.envs.registration import register
register(id="ChessEnv-v0", entry_point=__name__ + ":ChessEnv")
