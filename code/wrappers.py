# MIT License

# Copyright (c) 2020 FT Autonomous Team One

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

import gymnasium as gym
import numpy as np

from gymnasium import spaces

def convert_range(value, input_range, output_range):
    # 값(들)을 한 범위에서 다른 범위로 변환합니다
    # 범위 ---> [최소, 최대]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    Stable Baselines 3와 함께 사용하기 위한 F1Tenth Gym 환경용 래퍼.
    LiDAR 스캔만 관측값으로 반환하고 행동 정규화를 처리합니다.
    """

    def __init__(self, env):
        super().__init__(env)

        # 조향 및 속도에 대한 정규화된 행동 공간, 범위 [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # 관측 공간은 LiDAR 스캔 데이터(원시 거리)입니다
        self.lidar_max = 30.0  # LiDAR의 최대 범위
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.lidar_max,
            shape=(1080,),
            dtype=np.float32
        )

        # 행동 변환을 위한 조향 및 속도 범위 저장
        self.s_min = self.env.unwrapped.params['s_min']
        self.s_max = self.env.unwrapped.params['s_max']
        self.v_min = self.env.unwrapped.params['v_min']
        self.v_max = self.env.unwrapped.params['v_max']
        self.training_speed_limit = None  # 학습 중 속도 상한 (m/s), None이면 제한 없음
        self.training_speed_min = None   # 학습 중 속도 하한 (m/s), None이면 환경 기본 사용
        self._base_orientation_jitter = np.pi / 4  # direction 미제공 시 기본 지터
        self._aligned_orientation_jitter = 0.0     # direction 제공 시 적용할 지터 (기본 0)
        self.last_command_speed = 0.0

        # 리셋 로직을 위한 자동차 크기 및 트랙 정보 저장
        self.car_length = self.env.unwrapped.params['length']
        self.car_width = self.env.unwrapped.params['width']
        self.track_width = 3.2
        self.start_radius = (self.track_width / 2) - ((self.car_length + self.car_width) / 2)

        self.step_count = 0
        self.max_theta = 100 # 회전을 방지하기 위한 최대 자동차 각도 임계값
        self.last_reset_info = {}
        self.last_raw_observation = None

    def step(self, action):
        """Stable Baselines 호환 step. Gymnasium 반환값을 구 API 형태로 변환."""
        converted_action = self.un_normalise_actions(action)
        self.last_command_speed = converted_action[1]
        raw_observation, reward, terminated, truncated, info = self.env.step(
            np.array([converted_action])
        )

        self.step_count += 1
        self.last_raw_observation = raw_observation
        info = {} if info is None else dict(info)

        # 자동차가 스핀아웃하면 즉시 종료
        if abs(raw_observation['poses_theta'][0]) > self.max_theta:
            terminated = True
            info.setdefault('episode_ended_by', 'spinout')

        terminated = bool(terminated)
        truncated = bool(truncated)
        info['raw_observation'] = raw_observation
        info['terminated'] = terminated
        info['truncated'] = truncated

        lidar_scan = raw_observation['scans'][0].astype(np.float32)
        return lidar_scan, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None, start_xy=None, direction=None):
        if start_xy is None:
            start_xy = np.zeros(2)
        if direction is None:
            direction = np.random.uniform(0, 2 * np.pi)

        self.step_count = 0

        reset_options = {} if options is None else dict(options)
        if 'poses' not in reset_options:
            # 트랙에서 무작위 시작 위치 가져오기 (방향에 수직인 벡터 사용)
            lateral_dir = np.array([-np.sin(direction), np.cos(direction)])
            lateral_dir = lateral_dir / (np.linalg.norm(lateral_dir) + 1e-9)
            rand_offset = np.random.uniform(-1, 1) * self.start_radius
            spawn_xy = start_xy + rand_offset * lateral_dir

            jitter = self._aligned_orientation_jitter if direction is not None else self._base_orientation_jitter
            theta = np.random.uniform(-jitter, jitter) + direction
            reset_options['poses'] = np.array([[spawn_xy[0], spawn_xy[1], theta]], dtype=float)

        raw_observation, info = self.env.reset(seed=seed, options=reset_options)
        info = {} if info is None else dict(info)
        info['raw_observation'] = raw_observation
        info['terminated'] = False
        info['truncated'] = False

        self.last_reset_info = info
        self.last_raw_observation = raw_observation

        lidar_scan = raw_observation['scans'][0].astype(np.float32)
        return lidar_scan, info

    def un_normalise_actions(self, actions: np.ndarray) -> np.ndarray:
        """행동을 [-1, 1] 범위에서 조향/속도 범위로 변환합니다."""
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed_min = self.v_min if self.training_speed_min is None else max(0.0, self.training_speed_min)
        speed_max = self.v_max if self.training_speed_limit is None else min(self.v_max, self.training_speed_limit)
        if speed_min > speed_max:
            speed_min = speed_max
        speed = convert_range(actions[1], [-1, 1], [speed_min, speed_max])
        return np.array([steer, speed])

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed: int):
        self.env.seed(seed)
        np.random.seed(seed)
        print(f"F110_Wrapped seeded with {seed}")

    def set_training_speed_limit(self, limit: float | None):
        """
        학습 중 최대 속도를 설정합니다. limit=None이면 원래 차량 속도 한계를 사용합니다.
        """
        self.training_speed_limit = limit
        if limit is not None:
            self.training_speed_limit = max(0.0, limit)

    def set_spawn_orientation_jitter(self, jitter_aligned: float, jitter_default: float | None = None):
        """
        스폰 시 방향 지터를 설정합니다.

        :param jitter_aligned: direction 인자를 제공할 때 적용할 지터(라디안).
        :param jitter_default: direction 미제공 시 사용할 기본 지터(라디안). None이면 변경하지 않습니다.
        """
        self._aligned_orientation_jitter = max(jitter_aligned, 0.0)
        if jitter_default is not None:
            self._base_orientation_jitter = max(jitter_default, 0.0)

    def set_training_speed_min(self, min_speed: float | None):
        """
        학습 시 속도 하한을 설정합니다.
        """
        if min_speed is None:
            self.training_speed_min = None
        else:
            self.training_speed_min = max(0.0, min_speed)


class LidarNormalizeWrapper(gym.ObservationWrapper):
    """
    LiDAR 관측값을 실행 중 평균/분산으로 정규화하여 학습 안정성을 높입니다.
    """

    def __init__(self, env, epsilon: float = 1e-6, clip: float = 5.0):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("LidarNormalizeWrapper expects a Box observation space.")
        if env.observation_space.shape != (1080,):
            raise ValueError(
                f"LidarNormalizeWrapper expects observation shape (1080,), "
                f"but got {env.observation_space.shape}"
            )

        self.epsilon = epsilon
        self.clip = clip
        self.count = 0.0
        self.mean = np.zeros(1080, dtype=np.float64)
        self.m2 = np.ones(1080, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-clip,
            high=clip,
            shape=(1080,),
            dtype=np.float32
        )

        print("[LidarNormalizeWrapper] Running mean/var normalisation enabled.")

    def observation(self, observation):
        self._update_stats(observation)
        variance = self._variance()
        std = np.sqrt(variance + self.epsilon)
        normalized = (observation - self.mean) / std
        normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.astype(np.float32)

    def _update_stats(self, observation: np.ndarray):
        self.count += 1.0
        delta = observation - self.mean
        self.mean += delta / self.count
        delta2 = observation - self.mean
        self.m2 += delta * delta2

    def _variance(self):
        if self.count < 2:
            return np.ones_like(self.mean)
        return np.maximum(self.m2 / (self.count - 1.0), self.epsilon)
