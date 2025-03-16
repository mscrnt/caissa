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


logger = logging.getLogger(__name__)  # ✅ Ensure logging is available


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