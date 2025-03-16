# path: ./train.py

import os
import logging
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from custom_cnn import ChessCNN  # ✅ Import the custom CNN
from make_sb3_env import make_sb3_env  # ✅ Import our new function

# ✅ Ensure multiprocessing safety
if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)  # ✅ Ensure thread-safe multiprocessing

    # ✅ Linear schedule function
    def linear_schedule(initial_value, final_value=0.0):
        """Linear decay for parameters over training."""
        def func(progress):
            return final_value + progress * (initial_value - final_value)
        return func

    # ✅ Set up logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [Env %(process)d] %(message)s",
    )

    # ✅ Ensure multiprocessing starts safely
    num_envs = 16  # ✅ Run 16 games at once
    env, num_envs = make_sb3_env(num_envs=num_envs, seed=42, use_subprocess=True)

    n_steps = 256  
    n_epochs = 4  
    gamma = 0.99  
    gae_lambda = 0.95  
    ent_coef = 0.02  
    vf_coef = 0.5  
    max_grad_norm = 0.5  

    batch_size = (n_steps * num_envs) // 4  # ✅ Ensures 4 minibatches per update

    # ✅ Use linear schedule for learning rate and clip range
    learning_rate_schedule = linear_schedule(2.5e-4, 1e-5)
    clip_range_schedule = linear_schedule(0.1, 0.02)

    # ✅ Set random seed for reproducibility
    set_random_seed(42)

    # ✅ Initialize PPO model with MultiInputPolicy + Custom CNN
    model = PPO(
        "MultiInputPolicy",  
        env,
        verbose=1,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device="cuda",  
        tensorboard_log=log_dir,  
        n_steps=n_steps,  
        batch_size=batch_size,  
        n_epochs=n_epochs,  
        learning_rate=learning_rate_schedule,  
        clip_range=clip_range_schedule,  
        ent_coef=ent_coef,  
        vf_coef=vf_coef,  
        max_grad_norm=max_grad_norm,  
        policy_kwargs={
            "features_extractor_class": ChessCNN,  
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,  
        },
    )

    # ✅ Train for 16M steps with TensorBoard logging
    model.learn(total_timesteps=32_000_000)

    # ✅ Save the trained model
    model.save("chess_ppo_cnn_agent")
    print("Training completed and model saved!")
