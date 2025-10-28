# MIT 라이선스
#
# Formula Trinity Autonomous의 원본 training.py에서 수정됨
# 추가됨: LiDAR 기반 레이싱을 위한 SAC 알고리즘 + CNN 정책
#
# Copyright (c) 2021 Eoin Gogarty, Charlie Maguire and Manus McAuliffe (Formula Trintiy Autonomous)

"""
SAC + CNN 정책을 사용하는 F1Tenth Gym용 Stable Baselines 3 훈련 스크립트

원본 PPO+MLP 버전에 대한 주요 개선 사항:
- SAC 알고리즘 (연속 제어에 더 좋음)
- LiDAR 특징 추출을 위한 CNN 정책 (TinyLidarNet 아키텍처)
- 개선된 보상 형성 (트랙 진행, 위험 페널티, 속도 보상)
"""

import os
import gymnasium as gym
import f110_gym  # Register F1TENTH environments
import time
import glob
import wandb
import argparse
import numpy as np

from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from code.wrappers import F110_Wrapped
from code.improved_rewards import F110_ImprovedReward
from code.cnn_policy import CNNSACPolicy
from code.eoin_callbacks import SaveOnBestTrainingRewardCallback


# 훈련 구성
TRAIN_DIRECTORY = "./train_sac_cnn"
TRAIN_STEPS = pow(10, 5)  # 훈련 주기당 10만 스텝
NUM_PROCESS = 4  # 병렬 환경
MAP_PATH = "./f1tenth_racetracks/underground/underground_map"
MAP_EXTENSION = ".png"
TENSORBOARD_PATH = "./sac_cnn_tensorboard"
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 10)


def main(args):

    #       #
    #   훈련  #
    #       #

    # 가중치 및 편향 초기화
    if args.wandb:
        wandb.init(sync_tensorboard=True, project="f1tenth-sac-cnn")

    # 환경 준비
    def wrap_env():
        # F110 gym 시작
        env = gym.make("f110-v0",
                       map=MAP_PATH,
                       map_ext=MAP_EXTENSION,
                       num_agents=1)

        # 기본 RL 기능으로 래핑
        env = F110_Wrapped(env)

        # 개선된 보상 형성 적용 (Isaac Lab 스타일)
        env = F110_ImprovedReward(env, debug_mode=args.debug)

        # Underground 맵 고정 사용 - centerline 기반 진행거리 보상
        # RandomMap wrapper 제거됨 (맵 변경 없음)

        return env

    # 로그 디렉토리 생성
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(TRAIN_DIRECTORY, exist_ok=True)

    # 환경 벡터화 (병렬화)
    envs = make_vec_env(wrap_env,
                        n_envs=NUM_PROCESS,
                        seed=np.random.randint(pow(2, 32) - 1),
                        monitor_dir=log_dir,
                        vec_env_cls=SubprocVecEnv)

    # 모델 로드 또는 생성
    model, reset_num_timesteps = load_model(args.load,
                                            TRAIN_DIRECTORY,
                                            envs,
                                            TENSORBOARD_PATH)

    # 모델 저장 콜백 생성
    saving_callback = SaveOnBestTrainingRewardCallback(
        check_freq=SAVE_CHECK_FREQUENCY,
        log_dir=log_dir,
        save_dir=TRAIN_DIRECTORY,
        use_wandb=args.wandb,
        always_save=args.save
    )

    # 모델 훈련 및 소요 시간 기록
    start_time = time.time()
    model.learn(total_timesteps=TRAIN_STEPS,
                reset_num_timesteps=reset_num_timesteps,
                callback=saving_callback)

    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")
    print("Training cycle complete.")

    # 고유한 타임스탬프로 모델 저장
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model.save(f"{TRAIN_DIRECTORY}/sac-cnn-{timestamp}-final")
    if args.wandb:
        wandb.save(f"{TRAIN_DIRECTORY}/sac-cnn-{timestamp}-final.zip")


def load_model(load_arg, train_directory, envs, tensorboard_path=None, evaluating=False):
    """
    새로운 SAC+CNN 모델을 생성하거나 기존 체크포인트를 로드합니다.

    Args:
        load_arg: None (새 모델), "latest" (최신 체크포인트) 또는 특정 모델 이름
        train_directory: 저장된 모델이 포함된 디렉토리
        envs: 벡터화된 환경
        tensorboard_path: 텐서보드 로그 경로
        evaluating: True인 경우 평가용으로 로드 (훈련 없음)

    Returns:
        model: CNN 정책을 사용하는 SAC 모델
        reset_num_timesteps: 타임스텝 카운터를 리셋할지 여부
    """

    # 새 모델 생성
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

        # SAC 하이퍼파라미터 (F1TENTH 레이싱에 맞게 조정됨)
        model = SAC(
            policy=CNNSACPolicy,
            env=envs,
            learning_rate=3e-4,
            buffer_size=100000,  # 리플레이 버퍼 크기
            learning_starts=5000,  # 5000 스텝 후 훈련 시작
            batch_size=512,
            tau=0.005,  # 소프트 업데이트 계수
            gamma=0.99,  # 할인 계수
            train_freq=10,  # 매 스텝마다 훈련
            gradient_steps=1,
            ent_coef='auto',  # 자동 엔트로피 튜닝
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,  # 상태 종속 탐색 없음
            verbose=1,
            tensorboard_log=tensorboard_path,
            device='auto'  # 사용 가능한 경우 GPU 사용
        )

        print("SAC hyperparameters:")
        print(f"  - Learning rate: 3e-4")
        print(f"  - Buffer size: 100k")
        print(f"  - Batch size: 512")
        print(f"  - Gamma: 0.99")
        print(f"  - Entropy coefficient: auto")
        print("=" * 60)

    # 모델 로드
    else:
        reset_num_timesteps = False

        # 훈련된 모델 목록 가져오기
        trained_models = glob.glob(f"{train_directory}/*.zip")

        if not trained_models:
            print(f"No models found in {train_directory}. Creating new model...")
            return load_model(None, train_directory, envs, tensorboard_path, evaluating)

        # 최신 모델
        if (load_arg == "latest") or (load_arg is None):
            model_path = max(trained_models, key=os.path.getctime)
        else:
            trained_models_sorted = sorted(trained_models,
                                          key=os.path.getctime,
                                          reverse=True)
            # 사용자 입력을 모델 이름과 일치시킴
            matching_models = [m for m in trained_models_sorted if load_arg in m]
            if not matching_models:
                print(f"No model matching '{load_arg}' found. Using latest model...")
                model_path = max(trained_models, key=os.path.getctime)
            else:
                model_path = matching_models[0]

        # 출력을 위한 일반 모델 이름 가져오기
        model_name = os.path.basename(model_path).replace(".zip", '')
        print("=" * 60)
        print(f"Loading model: {model_name}")
        print(f"From: {train_directory}")
        print("=" * 60)

        # 경로에서 모델 로드
        model = SAC.load(model_path)

        # 환경 설정 및 리셋
        model.set_env(envs)
        envs.reset()

        print("Model loaded successfully!")
        print("=" * 60)

    return model, reset_num_timesteps


# Python 다중 처리에 필요
if __name__ == "__main__":
    # 스크립트에 대한 런타임 인수 구문 분석
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

    # 주 훈련 함수 호출
    main(args)