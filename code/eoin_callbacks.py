import os

import time
import wandb
import numpy as np

from datetime import datetime

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.lr_scheduler import CosineAnnealingLR


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


class LearningRateSchedulerCallback(BaseCallback):
    """
    학습률 스케줄러를 관리하고 TensorBoard에 모든 학습률을 로깅하는 콜백.

    이 콜백은:
    1. 각 학습 스텝마다 모든 optimizer의 스케줄러를 업데이트
    2. 현재 학습률을 TensorBoard에 로깅 (Actor CNN/MLP, Critic CNN/MLP, Entropy)

    :param schedulers: dict, 스케줄러 딕셔너리 {'actor': scheduler, 'critic': scheduler, 'entropy': scheduler}
    :param log_freq: int, 로깅 빈도 (스텝 단위)
    :param verbose: int, 상세 출력 레벨
    """

    def __init__(self, schedulers: dict, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.schedulers = schedulers
        self.log_freq = log_freq
        self.last_log = 0

    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 로깅 시간인지 확인
        if self.num_timesteps >= self.last_log + self.log_freq:
            self.last_log = self.num_timesteps

            # Actor 학습률 로깅 (CNN과 MLP 구분)
            if 'actor' in self.schedulers and self.schedulers['actor'] is not None:
                actor_optimizer = self.model.actor.optimizer
                param_groups = actor_optimizer.param_groups

                if len(param_groups) >= 2:
                    # CNN과 MLP가 분리되어 있는 경우
                    actor_cnn_lr = param_groups[0]['lr']
                    actor_mlp_lr = param_groups[1]['lr']
                    self.logger.record("train/actor_cnn_lr", actor_cnn_lr)
                    self.logger.record("train/actor_mlp_lr", actor_mlp_lr)
                elif len(param_groups) == 1:
                    # 단일 파라미터 그룹
                    actor_lr = param_groups[0]['lr']
                    self.logger.record("train/actor_lr", actor_lr)

            # Critic 학습률 로깅 (CNN과 MLP 구분)
            if 'critic' in self.schedulers and self.schedulers['critic'] is not None:
                critic_optimizer = self.model.critic.optimizer
                param_groups = critic_optimizer.param_groups

                if len(param_groups) >= 2:
                    # CNN과 MLP가 분리되어 있는 경우
                    critic_cnn_lr = param_groups[0]['lr']
                    critic_mlp_lr = param_groups[1]['lr']
                    self.logger.record("train/critic_cnn_lr", critic_cnn_lr)
                    self.logger.record("train/critic_mlp_lr", critic_mlp_lr)
                elif len(param_groups) == 1:
                    # 단일 파라미터 그룹
                    critic_lr = param_groups[0]['lr']
                    self.logger.record("train/critic_lr", critic_lr)

            # Entropy 학습률 로깅
            if 'entropy' in self.schedulers and self.schedulers['entropy'] is not None:
                entropy_optimizer = self.model.log_ent_coef_optimizer
                entropy_lr = entropy_optimizer.param_groups[0]['lr']
                self.logger.record("train/entropy_lr", entropy_lr)

        return True

    def _on_rollout_end(self) -> None:
        """Rollout이 끝날 때마다 스케줄러 업데이트"""
        # Actor 스케줄러 스텝
        if 'actor' in self.schedulers and self.schedulers['actor'] is not None:
            self.schedulers['actor'].step()

        # Critic 스케줄러 스텝
        if 'critic' in self.schedulers and self.schedulers['critic'] is not None:
            self.schedulers['critic'].step()

        # Entropy 스케줄러 스텝
        if 'entropy' in self.schedulers and self.schedulers['entropy'] is not None:
            self.schedulers['entropy'].step()
