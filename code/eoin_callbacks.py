import os

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
                    model_filename = f"sac-{timestamp}-{int(mean_reward)}R.zip"

                    # 1. 전체 모델 저장 (학습 재개용 - optimizer 포함)
                    self.model.save(f"{self.save_path}/{model_filename}")

                    # 2. 추론용 모델 저장 (policy만 - ROS2 배포용)
                    # optimizer를 제외하고 저장하여 로딩 호환성 확보
                    inference_filename = f"sac-{timestamp}-{int(mean_reward)}R-inference.zip"
                    try:
                        self.model.save(f"{self.save_path}/{inference_filename}",
                                       exclude=['replay_buffer'])
                        if self.verbose > 0:
                            print(f"  추론용 모델 저장: {inference_filename}")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"  추론용 모델 저장 실패: {e}")

                    if self.use_wandb:
                        wandb.save(f"{self.save_path}/{model_filename}")
                        wandb.save(f"{self.save_path}/{inference_filename}")

        return True


class LearningRateSchedulerCallback(BaseCallback):
    """
    progress_remaining(=1→0) 기반 스케줄을 적용하며 현재 학습률을 로깅하는 콜백.

    :param schedule_cfg: {'actor': {'schedule': fn, 'ratios': [...]}, 'critic': {...}, 'entropy': {'schedule': fn}}
    :param log_freq: TensorBoard 로깅 주기 (스텝 기준)
    """

    def __init__(self, schedule_cfg: dict | None = None, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.schedule_cfg = schedule_cfg or {}
        self.log_freq = log_freq
        self.last_log = 0

    def _record_optimizer_lrs(self, prefix: str, optimizer) -> None:
        param_groups = optimizer.param_groups
        if len(param_groups) >= 2:
            self.logger.record(f"train/{prefix}_cnn_lr", param_groups[0]['lr'])
            self.logger.record(f"train/{prefix}_mlp_lr", param_groups[1]['lr'])
        elif len(param_groups) == 1:
            self.logger.record(f"train/{prefix}_lr", param_groups[0]['lr'])

    def _apply_schedule(self, key: str, optimizer, progress: float) -> None:
        cfg = self.schedule_cfg.get(key)
        if not cfg or optimizer is None:
            return

        schedule_fn = cfg.get("schedule")
        if schedule_fn is None:
            return

        base_lr = schedule_fn(progress)
        ratios = cfg.get("ratios")
        if ratios:
            for group, ratio in zip(optimizer.param_groups, ratios):
                group['lr'] = base_lr * ratio
        else:
            for group in optimizer.param_groups:
                group['lr'] = base_lr

    def _on_step(self) -> bool:
        progress = getattr(self.model, "_current_progress_remaining", 1.0)

        if hasattr(self.model, "actor") and self.model.actor is not None:
            self._apply_schedule("actor", self.model.actor.optimizer, progress)

        if hasattr(self.model, "critic") and self.model.critic is not None:
            self._apply_schedule("critic", self.model.critic.optimizer, progress)

        if hasattr(self.model, "log_ent_coef_optimizer") and self.model.log_ent_coef_optimizer is not None:
            self._apply_schedule("entropy", self.model.log_ent_coef_optimizer, progress)

        if self.num_timesteps >= self.last_log + self.log_freq:
            self.last_log = self.num_timesteps

            if hasattr(self.model, "actor") and self.model.actor is not None:
                self._record_optimizer_lrs("actor", self.model.actor.optimizer)
                self.logger.record("train/learning_rate", self.model.actor.optimizer.param_groups[-1]['lr'])

            if hasattr(self.model, "critic") and self.model.critic is not None:
                self._record_optimizer_lrs("critic", self.model.critic.optimizer)

            if hasattr(self.model, "log_ent_coef_optimizer") and self.model.log_ent_coef_optimizer is not None:
                entropy_lr = self.model.log_ent_coef_optimizer.param_groups[0]['lr']
                self.logger.record("train/entropy_lr", entropy_lr)

        return True
