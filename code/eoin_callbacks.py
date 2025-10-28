import os

import gym
import time
import wandb
import numpy as np

from datetime import datetime

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    훈련 보상을 기반으로 모델을 저장하기 위한 콜백 (``check_freq`` 스텝마다 확인).
    (실제로는 ``EvalCallback`` 사용을 권장합니다).

    :param check_freq: (int)
    :param log_dir: (str) 모델이 저장될 폴더의 경로.
      ``Monitor`` 래퍼에 의해 생성된 파일을 포함해야 합니다.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, save_dir: str, use_wandb: bool, always_save=False, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_dir
        self.best_mean_reward = -np.inf
        self.use_wandb = use_wandb
        self.always_save = always_save
        # 가장 최근 저장에 대한 스텝 카운터
        self.last_check = 0

    def _init_callback(self) -> None:
        # 필요한 경우 폴더 생성
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > (self.last_check + self.check_freq):

            # 마지막 확인 카운터 업데이트
            self.last_check = self.num_timesteps

            # 훈련 보상 검색
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # 지난 100개 에피소드에 대한 평균 훈련 보상
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # 새로운 최고 모델, 여기에 에이전트를 저장할 수 있습니다
                if (mean_reward > self.best_mean_reward) or self.always_save:
                    self.best_mean_reward = mean_reward
                    # 최고 모델 저장 예시
                    if self.verbose > 0:
                        if self.always_save:
                            print("Saving current model to {}".format(self.save_path))
                        else:
                            print("Saving new best model to {}".format(self.save_path))
                    # 고유한 타임스탬프로 모델 저장
                    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    self.model.save(
                        f"{self.save_path}/sac-{timestamp}-{int(mean_reward)}R.zip")
                    if self.use_wandb:
                        wandb.save(
                            f"{self.save_path}/sac-{timestamp}-{int(mean_reward)}R.zip")

        return True