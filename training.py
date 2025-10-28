"""
SAC와 사용자 정의 CNN 정책을 사용하는 F1Tenth Gym용 Stable Baselines 3 훈련 스크립트.
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

from code.wrappers import F110_Wrapped, RandomF1TenthMap
from code.cnn_policy import LidarFeatureExtractor
from code.eoin_callbacks import SaveOnBestTrainingRewardCallback

# 스크립트 상수
TRAIN_DIRECTORY = "./train_sac_cnn"
TRAIN_STEPS = 500_000
NUM_PROCESS = 4
MAP_CHANGE_INTERVAL = 5000
TENSORBOARD_PATH = "./sac_cnn_tensorboard"
SAVE_CHECK_FREQUENCY = 5000

def main(args):
    """메인 훈련 루틴."""

    if args.wandb:
        wandb.init(sync_tensorboard=True, project="F1Tenth-RL-SAC")

    # 로그 디렉토리 생성
    os.makedirs(TRAIN_DIRECTORY, exist_ok=True)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # 환경 설정 함수
    def wrap_env():
        env = gym.make("f110_gym:f110-v0", num_agents=1)
        # 래퍼 적용
        env = F110_Wrapped(env)
        env = RandomF1TenthMap(env, step_interval=MAP_CHANGE_INTERVAL)
        return env

    # 벡터화된 환경 생성
    envs = make_vec_env(
        wrap_env,
        n_envs=NUM_PROCESS,
        seed=np.random.randint(pow(2, 32) - 1),
        monitor_dir=log_dir,
        vec_env_cls=SubprocVecEnv
    )

    # 사용자 정의 CNN 특징 추출기를 위한 정책 키워드 인수 정의
    # SAC 액터/크리틱 네트워크에 표준 아키텍처 사용
    policy_kwargs = dict(
        features_extractor_class=LidarFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    # SAC 모델 로드 또는 생성
    model, reset_num_timesteps = load_model(
        args.load,
        envs,
        policy_kwargs,
        TENSORBOARD_PATH
    )

    # 최적 모델 저장을 위한 콜백
    saving_callback = SaveOnBestTrainingRewardCallback(
        check_freq=SAVE_CHECK_FREQUENCY,
        log_dir=log_dir,
        save_dir=TRAIN_DIRECTORY,
        use_wandb=args.wandb,
        always_save=args.save
    )

    # 모델 훈련
    print("--- Training SAC with CNN Policy ---")
    start_time = time.time()
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=reset_num_timesteps,
        callback=saving_callback
    )
    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 최종 모델 저장
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    final_model_path = f"{TRAIN_DIRECTORY}/sac-cnn-{timestamp}-final"
    model.save(final_model_path)
    if args.wandb:
        wandb.save(f"{final_model_path}.zip")

    print("Training cycle complete.")

def load_model(load_arg, envs, policy_kwargs, tensorboard_path):
    """
    파일에서 SAC 모델을 로드하거나 새로 생성합니다.
    """
    if load_arg:
        reset_num_timesteps = False
        if load_arg == "latest":
            model_list = glob.glob(f"{TRAIN_DIRECTORY}/*.zip")
            model_path = max(model_list, key=os.path.getctime)
        else:
            model_path = load_arg
        
        print(f"Loading model from: {model_path}")
        model = SAC.load(model_path, env=envs)
        
    else:
        print("Creating new SAC model...")
        reset_num_timesteps = True
        model = SAC(
            "MlpPolicy",
            envs,
            policy_kwargs=policy_kwargs,
            verbose=1,
            buffer_size=100_000,       # 리플레이 버퍼 크기
            learning_rate=3e-4,         # skrl_sac_cfg.yaml에 해당
            batch_size=256,             # skrl_sac_cfg.yaml에 해당
            gamma=0.99,                 # 할인 계수
            tau=0.005,                  # Polyak 업데이트 계수
            train_freq=(1, "step"),     # 각 스텝 후 훈련
            learning_starts=10000,      # 1만 스텝 후 학습 시작
            tensorboard_log=tensorboard_path
        )

    return model, reset_num_timesteps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an F1Tenth agent using SAC with a CNN policy.")
    parser.add_argument("-l", "--load", help="Load a trained model (path or 'latest')", nargs="?", const="latest")
    parser.add_argument("-w", "--wandb", help="Use Weights and Biases for logging", action="store_true")
    parser.add_argument("-s", "--save", help="Always save model at callback frequency", action="store_true")
    args = parser.parse_args()
    
    main(args)