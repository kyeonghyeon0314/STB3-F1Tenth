# MIT 라이선스

# Copyright (c) 2021 Eoin Gogarty, Charlie Maguire and Manus McAuliffe (Formula Trintiy Autonomous)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
래핑된 환경을 사용하는 F1Tenth Gym용 Stable Baselines 3 평가 스크립트
"""

import os
import gymnasium as gym
import time
import glob
import argparse
import numpy as np

from datetime import datetime

from stable_baselines3 import PPO

from code.wrappers import F110_Wrapped, RandomMap


TRAIN_DIRECTORY = "./train"
MIN_EVAL_EPISODES = 5
MAP_PATH = "./f1tenth_racetracks/underground/underground_map"
MAP_EXTENSION = ".png"
MAP_CHANGE_INTERVAL = 3000


def main(args):

    #          #
    #   평가   #
    #          #

    # 환경 준비
    def wrap_env():
        # F110 gym 시작
        env = gym.make("f110_gym:f110-v0",
                       map=MAP_PATH,
                       map_ext=MAP_EXTENSION,
                       num_agents=1)
        # RL 함수로 기본 gym 래핑
        env = F110_Wrapped(env)
        env = RandomMap(env, MAP_CHANGE_INTERVAL)
        return env

    # 평가 환경 생성 (훈련 환경과 동일)
    eval_env = wrap_env()

    # 무작위 시드 설정
    eval_env.seed(np.random.randint(pow(2, 32) - 1))

    # 모델 로드 또는 생성
    model, _ = load_model(args.load,
                          TRAIN_DIRECTORY,
                          eval_env,
                          evaluating=True)

    # 몇 개의 에피소드를 시뮬레이션하고 렌더링, 에피소드를 취소하려면 ctrl-c
    episode = 0
    while episode < MIN_EVAL_EPISODES:
        try:
            episode += 1
            reset_result = eval_env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            done = False
            while not done:
                # 훈련된 모델을 사용하여 관측값을 기반으로 행동 예측
                action, _ = model.predict(obs)
                step_result = eval_env.step(action)
                if isinstance(step_result, tuple) and len(step_result) == 5:
                    obs, _, terminated, truncated, _ = step_result
                    done = bool(terminated or truncated)
                else:
                    obs, _, done, _ = step_result
                eval_env.render()
            # 이 섹션은 사용자에게 더 많은 에피소드를 실행할지 묻습니다
            if episode == (MIN_EVAL_EPISODES - 1):
                choice = input("Another episode? (Y/N) ")
                if choice.replace(" ", "").lower() in ["y", "yes"]:
                    episode -= 1
                else:
                    episode = MIN_EVAL_EPISODES
        except KeyboardInterrupt:
            pass


def load_model(load_arg, train_directory, envs, tensorboard_path=None, evaluating=False):
    '''
    아래 "새 모델 생성" 섹션에 지정된 대로 새 모델을 생성하거나,
    가장 최근에 훈련된 모델(또는 사용자가 지정한 모델)을 로드하여
    훈련을 계속하는 약간 복잡한 함수
    '''
    
    # 새 모델 생성
    if (load_arg is None) and (not evaluating):
        print("Creating new model...")
        reset_num_timesteps = True
        model = PPO("MlpPolicy",
                    envs,
                    verbose=1,
                    tensorboard_log=tensorboard_path)
    # 모델 로드
    else:
        reset_num_timesteps = False
        # 훈련된 모델 목록 가져오기
        trained_models = glob.glob(f"{train_directory}/*")
        # 최신 모델
        if (load_arg == "latest") or (load_arg is None):
            model_path = max(trained_models, key=os.path.getctime)
        else:
            trained_models_sorted = sorted(trained_models,
                                           key=os.path.getctime,
                                           reverse=True)
            # 사용자 입력을 모델 이름과 일치시킴
            model_path = [m for m in trained_models_sorted if load_arg in m]
            model_path = model_path[0]
        # 출력을 위한 일반 모델 이름 가져오기
        model_name = model_path.replace(".zip", '')
        model_name = model_name.replace(f"{train_directory}/", '')
        print(f"Loading model ({train_directory}) {model_name}")
        # 경로에서 모델 로드
        model = PPO.load(model_path)
        # 환경 설정 및 리셋
        model.set_env(envs)
        envs.reset()
    # 새/로드된 모델 반환
    return model, reset_num_timesteps


# Python 다중 처리에 필요 (평가에는 필요 없음)
if __name__ == "__main__":
    # 스크립트에 대한 런타임 인수 구문 분석
    parser = argparse.ArgumentParser()
    parser.add_argument("-l",
                        "--load",
                        help="load previous model",
                        nargs="?",
                        const="latest")
    args = parser.parse_args()
    # 주 훈련 함수 호출
    main(args)
