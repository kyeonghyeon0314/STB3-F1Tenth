# MIT License
#
# Modified from original training.py by Formula Trinity Autonomous
# Added: SAC algorithm + CNN policy for LiDAR-based racing
#
# Copyright (c) 2021 Eoin Gogarty, Charlie Maguire and Manus McAuliffe (Formula Trintiy Autonomous)

"""
Stable Baselines 3 training script for F1Tenth Gym with SAC + CNN Policy

Key improvements over original PPO+MLP version:
- SAC algorithm (better for continuous control)
- CNN policy for LiDAR feature extraction (TinyLidarNet architecture)
- Improved reward shaping (track progress, danger penalties, speed rewards)
"""

import os
import gym
import time
import glob
import wandb
import argparse
import numpy as np

from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from code.wrappers import F110_Wrapped, RandomMap
from code.improved_rewards import F110_ImprovedReward
from code.cnn_policy import CNNSACPolicy
from code.eoin_callbacks import SaveOnBestTrainingRewardCallback


# Training configuration
TRAIN_DIRECTORY = "./train_sac_cnn"
TRAIN_STEPS = pow(10, 5)  # 100k steps per training cycle
NUM_PROCESS = 4  # Parallel environments
MAP_PATH = "./f1tenth_gym_ros/examples/example_map"
MAP_EXTENSION = ".png"
MAP_CHANGE_INTERVAL = 3000  # Change map every 3000 steps
TENSORBOARD_PATH = "./sac_cnn_tensorboard"
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 10)


def main(args):

    #       #
    # TRAIN #
    #       #

    # Initialize weights and biases
    if args.wandb:
        wandb.init(sync_tensorboard=True, project="f1tenth-sac-cnn")

    # Prepare the environment
    def wrap_env():
        # Start F110 gym
        env = gym.make("f110_gym:f110-v0",
                       map=MAP_PATH,
                       map_ext=MAP_EXTENSION,
                       num_agents=1)

        # Wrap with basic RL functions
        env = F110_Wrapped(env)

        # Apply improved reward shaping (Isaac Lab-style)
        env = F110_ImprovedReward(env, debug_mode=args.debug)

        # Random map generation
        env = RandomMap(env, MAP_CHANGE_INTERVAL)

        return env

    # Create log directory
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(TRAIN_DIRECTORY, exist_ok=True)

    # Vectorize environment (parallelize)
    envs = make_vec_env(wrap_env,
                        n_envs=NUM_PROCESS,
                        seed=np.random.randint(pow(2, 32) - 1),
                        monitor_dir=log_dir,
                        vec_env_cls=SubprocVecEnv)

    # Load or create model
    model, reset_num_timesteps = load_model(args.load,
                                            TRAIN_DIRECTORY,
                                            envs,
                                            TENSORBOARD_PATH)

    # Create the model saving callback
    saving_callback = SaveOnBestTrainingRewardCallback(
        check_freq=SAVE_CHECK_FREQUENCY,
        log_dir=log_dir,
        save_dir=TRAIN_DIRECTORY,
        use_wandb=args.wandb,
        always_save=args.save
    )

    # Train model and record time taken
    start_time = time.time()
    model.learn(total_timesteps=TRAIN_STEPS,
                reset_num_timesteps=reset_num_timesteps,
                callback=saving_callback)

    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")
    print("Training cycle complete.")

    # Save model with unique timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model.save(f"{TRAIN_DIRECTORY}/sac-cnn-{timestamp}-final")
    if args.wandb:
        wandb.save(f"{TRAIN_DIRECTORY}/sac-cnn-{timestamp}-final.zip")


def load_model(load_arg, train_directory, envs, tensorboard_path=None, evaluating=False):
    """
    Create new SAC+CNN model or load existing checkpoint.

    Args:
        load_arg: None (new model), "latest" (latest checkpoint), or specific model name
        train_directory: directory containing saved models
        envs: vectorized environments
        tensorboard_path: path for tensorboard logs
        evaluating: if True, load for evaluation (no training)

    Returns:
        model: SAC model with CNN policy
        reset_num_timesteps: whether to reset timestep counter
    """

    # Create new model
    if (load_arg is None) and (not evaluating):
        print("=" * 60)
        print("Creating new SAC model with CNN policy...")
        print("=" * 60)
        print("Architecture:")
        print("  - LiDAR (1080) → 1D CNN → 64 features")
        print("  - Actor: 64 → 64 → 64 → actions (mean, log_std)")
        print("  - Critic: 64 + actions → 128 → 128 → 64 → Q-value")
        print("=" * 60)

        reset_num_timesteps = True

        # SAC hyperparameters (tuned for F1TENTH racing)
        model = SAC(
            policy=CNNSACPolicy,
            env=envs,
            learning_rate=3e-4,
            buffer_size=100000,  # Replay buffer size
            learning_starts=1000,  # Start training after 1000 steps
            batch_size=256,
            tau=0.005,  # Soft update coefficient
            gamma=0.99,  # Discount factor
            train_freq=1,  # Train every step
            gradient_steps=1,
            ent_coef='auto',  # Automatic entropy tuning
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,  # No state-dependent exploration
            verbose=1,
            tensorboard_log=tensorboard_path,
            device='auto'  # Use GPU if available
        )

        print("SAC hyperparameters:")
        print(f"  - Learning rate: 3e-4")
        print(f"  - Buffer size: 100k")
        print(f"  - Batch size: 256")
        print(f"  - Gamma: 0.99")
        print(f"  - Entropy coefficient: auto")
        print("=" * 60)

    # Load model
    else:
        reset_num_timesteps = False

        # Get trained model list
        trained_models = glob.glob(f"{train_directory}/*.zip")

        if not trained_models:
            print(f"No models found in {train_directory}. Creating new model...")
            return load_model(None, train_directory, envs, tensorboard_path, evaluating)

        # Latest model
        if (load_arg == "latest") or (load_arg is None):
            model_path = max(trained_models, key=os.path.getctime)
        else:
            trained_models_sorted = sorted(trained_models,
                                          key=os.path.getctime,
                                          reverse=True)
            # Match user input to model names
            matching_models = [m for m in trained_models_sorted if load_arg in m]
            if not matching_models:
                print(f"No model matching '{load_arg}' found. Using latest model...")
                model_path = max(trained_models, key=os.path.getctime)
            else:
                model_path = matching_models[0]

        # Get plain model name for printing
        model_name = os.path.basename(model_path).replace(".zip", '')
        print("=" * 60)
        print(f"Loading model: {model_name}")
        print(f"From: {train_directory}")
        print("=" * 60)

        # Load model from path
        model = SAC.load(model_path)

        # Set and reset environment
        model.set_env(envs)
        envs.reset()

        print("Model loaded successfully!")
        print("=" * 60)

    return model, reset_num_timesteps


# Necessary for Python multi-processing
if __name__ == "__main__":
    # Parse runtime arguments to script
    parser = argparse.ArgumentParser(description="Train F1TENTH agent with SAC + CNN policy")
    parser.add_argument("-l", "--load",
                        help="load previous model (default: latest)",
                        nargs="?",
                        const="latest")
    parser.add_argument("-w", "--wandb",
                        help="use Weights and Biases API",
                        action="store_true")
    parser.add_argument("-s", "--save",
                        help="always save at step interval",
                        action="store_true")
    parser.add_argument("-d", "--debug",
                        help="enable debug logging",
                        action="store_true")
    args = parser.parse_args()

    # Call main training function
    main(args)
