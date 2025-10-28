"""
F1TENTH 레이싱을 위한 개선된 보상 함수

Isaac Lab의 고급 보상 형성에서 포팅됨:
- 트랙 진행 상황 추적 (중심선 투영)
- 연속적인 충돌 감지
- 속도 보상 및 저속 페널티
- 지수적 위험 페널티 (벽 근접성)
- 멈춤 감지

작성자: Isaac Lab f1tenth_env.py에서 포팅됨
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os


class F110_ImprovedReward(gym.Wrapper):
    """
    개선된 보상 형성을 갖춘 F1TENTH용 고급 보상 래퍼.

    특징:
    - 중심선 투영을 통한 트랙 진행 상황 추적
    - 속도 기반 전진 보상
    - 벽 근접에 대한 지수적 위험 페널티
    - 연속적인 LiDAR 기반 충돌 감지
    - 정지를 방지하기 위한 저속 페널티
    """

    def __init__(self, env, centerline_path=None, debug_mode=False):
        super().__init__(env)

        self.debug_mode = debug_mode

        # 진행 상황 추적을 위한 중심선 웨이포인트 로드
        if centerline_path is None:
            # 기본적으로 현재 맵의 중심선 사용
            map_path = env.unwrapped.map_name
            # map_path에서 맵 이름 추출 (예: .../underground_map -> underground)
            map_name = os.path.basename(map_path).replace('_map', '')
            centerline_path = f"./f1tenth_racetracks/{map_name}/{map_name}_centerline.csv"

        self.centerline_waypoints = np.genfromtxt(centerline_path, delimiter=',')[:, :2]  # X, Y만
        self._init_centerline()

        # 진행 상황 상태 추적
        self.current_waypoint_idx = 0
        self.previous_progress = 0.0
        self.total_distance = 0.0
        self.lap_count = 0
        self.previous_pos = np.zeros(2)

        # 멈춤 감지
        self.stuck_check_interval = 120  # 스텝 (60Hz에서 2초)
        self.stuck_threshold = 0.1  # 미터 (2초에 10cm = 평균 0.05 m/s)
        self.stuck_initial_delay = 300  # 스텝 (5초)
        self.last_check_pos = np.zeros(2)
        self.steps_since_last_check = 0
        self.command_slow_threshold = 0.1  # m/s
        self.command_slow_limit = 180  # 스텝
        self.command_slow_steps = 0

        # 에피소드 통계
        self.episode_step = 0
        self.episode_reward_sum = 0.0
        self.episode_forward_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_survival_sum = 0.0
        self.episode_danger_sum = 0.0
        self.episode_base_sum = 0.0
        self.command_slow_steps = 0

        # 속도 정규화를 위한 최대 속도 추출 (학습 시 속도 제한 우선)
        training_speed_limit = getattr(env, "training_speed_limit", None)
        if training_speed_limit is not None and training_speed_limit > 0:
            self.max_speed = training_speed_limit
        else:
            self.max_speed = getattr(env, "v_max", 5.0)
        self.target_speed = max(0.2 * self.max_speed, 0.1)

        print(f"[ImprovedReward] Initialized with centerline: {centerline_path}")
        print(f"  - Track length: {self.track_length:.2f}m")
        print(f"  - Num waypoints: {len(self.centerline)}")
        print(f"  - Speed normalization max: {self.max_speed:.2f} m/s (target {self.target_speed:.2f} m/s)")

    def _init_centerline(self):
        """
        부드러운 진행 상황 추적을 위해 중심선을 초기화하고 웨이포인트를 보간합니다.
        """
        waypoints = self.centerline_waypoints

        # 웨이포인트 간 선형 보간 (세그먼트당 100개 지점 = 1cm 해상도)
        interpolated_points = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            for j in range(100):
                t = j / 100.0
                point = start + t * (end - start)
                interpolated_points.append(point)

        # numpy 배열로 변환
        self.centerline = np.array(interpolated_points, dtype=np.float32)

        # 중심선을 따른 누적 거리 계산
        distances = np.linalg.norm(self.centerline[1:] - self.centerline[:-1], axis=1)
        self.centerline_cumulative_dist = np.concatenate([
            np.zeros(1),
            np.cumsum(distances)
        ])

        # 총 트랙 길이
        self.track_length = self.centerline_cumulative_dist[-1]

    def _get_track_progress(self, pos: np.ndarray) -> tuple:
        """
        세그먼트 투영을 통해 트랙 진행 거리를 계산합니다.

        차량 위치를 가장 가까운 중심선 세그먼트에 투영하여
        연속적인 진행 측정을 얻습니다 (자동차가 중심선에서 벗어난 경우 처리).

        Args:
            pos: 차량 XY 위치 (2,)

        Returns:
            progress: 트랙 진행 거리 [0, track_length]
            progress_delta: 마지막 스텝 이후 진행 상황 변화
        """
        # 세그먼트 투영 방법
        num_segments = len(self.centerline) - 1

        segment_starts = self.centerline[:-1]  # (세그먼트 수, 2)
        segment_ends = self.centerline[1:]     # (세그먼트 수, 2)
        segment_vectors = segment_ends - segment_starts
        segment_len_sq = np.sum(segment_vectors ** 2, axis=1)

        # 세그먼트 시작점에서 차량까지의 벡터
        to_vehicle = pos[np.newaxis, :] - segment_starts  # (세그먼트 수, 2)

        # 투영 비율 t = (to_vehicle · segment_vector) / ||segment_vector||^2
        dot_product = np.sum(to_vehicle * segment_vectors, axis=1)
        t = dot_product / (segment_len_sq + 1e-8)
        t = np.clip(t, 0.0, 1.0)  # 세그먼트에 고정

        # 투영된 지점
        projected_points = segment_starts + t[:, np.newaxis] * segment_vectors

        # 투영까지의 거리
        dist_to_proj = np.linalg.norm(pos[np.newaxis, :] - projected_points, axis=1)

        # 가장 가까운 세그먼트 찾기
        closest_seg = np.argmin(dist_to_proj)
        t_closest = t[closest_seg]

        # 진행 거리 = 세그먼트 시작 거리 + (t × 세그먼트 길이)
        seg_start_dist = self.centerline_cumulative_dist[closest_seg]
        seg_length = np.sqrt(segment_len_sq[closest_seg])
        progress = seg_start_dist + t_closest * seg_length

        # 결승선 통과 감지 (90% → 10% 전환)
        finish_line_crossed = (self.previous_progress > self.track_length * 0.9) and \
                              (progress < self.track_length * 0.1)

        if finish_line_crossed:
            self.lap_count += 1

        # 절대 진행 상황 (랩 전체에 걸쳐 연속적)
        absolute_progress = progress + self.lap_count * self.track_length

        # 진행 델타
        progress_delta = absolute_progress - self.total_distance

        # 상태 업데이트
        self.total_distance = absolute_progress
        self.current_waypoint_idx = closest_seg
        self.previous_progress = progress

        return progress, progress_delta

    def _get_centerline_direction(self) -> np.ndarray:
        """
        현재 위치에서 중심선 방향 벡터를 가져옵니다.

        Returns:
            direction: 정규화된 XY 방향 벡터 (2,)
        """
        segment_idx = self.current_waypoint_idx
        segment_start = self.centerline[segment_idx]
        segment_end = self.centerline[segment_idx + 1]

        direction = segment_end - segment_start
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        return direction

    def step(self, action):
        """
        개선된 보상 형성으로 스텝을 진행합니다.
        """
        # 원래 스텝 (LiDAR 관측 + 보조 정보)
        lidar_scan, base_reward, terminated, truncated, info = self.env.step(action)
        info = {} if info is None else dict(info)
        raw_observation = info.get('raw_observation')
        if raw_observation is None:
            raise KeyError("raw_observation missing from inner env info")

        self.episode_step += 1
        self.steps_since_last_check += 1

        # 상태 추출
        pos_xy = np.array([raw_observation['poses_x'][0], raw_observation['poses_y'][0]])
        vel_xy = np.array([
            raw_observation['linear_vels_x'][0],
            raw_observation['linear_vels_y'][0]
        ])
        lidar_scan = raw_observation['scans'][0]
        command_speed = abs(getattr(self.env, 'last_command_speed', 0.0))
        if command_speed <= self.command_slow_threshold:
            self.command_slow_steps += 1
        else:
            self.command_slow_steps = 0

        # --- 보상 계산 ---

        # 1) 트랙 진행 보상 (주요 목표)
        _, progress_delta = self._get_track_progress(pos_xy)
        FORWARD_SCALE = 5.0  # 12.0 → 5.0 (덜 지배적)
        reward_forward = np.clip(progress_delta, 0.0, None) * FORWARD_SCALE

        # 2) 속도 보상 (전체 속도 크기 - 빠른 주행 장려)
        actual_speed = np.linalg.norm(vel_xy)  # 방향 무관 전체 속도
        forward_direction = self._get_centerline_direction()
        forward_speed = np.sum(vel_xy * forward_direction)  # 방향 고려 속도 (페널티용)

        SPEED_SCALE = 3.0  # 0.6 → 3.0 (중요도 증가)
        SPEED_NORM = 5.0   # 실제 주행 속도 기준 (50.8 아님!)
        reward_speed = np.clip(actual_speed / SPEED_NORM, 0.0, 1.0) * SPEED_SCALE

        # 3) 저속 페널티/보상: 실질 전진이 거의 없을 때만 소폭 페널티를 주고, 충분히 움직이면 0으로 유지
        TARGET_SPEED = max(self.target_speed, 1e-3)
        MIN_PROGRESS = 0.05  # m
        if progress_delta < MIN_PROGRESS and forward_speed < TARGET_SPEED:
            penalty_ratio = (TARGET_SPEED - forward_speed) / TARGET_SPEED
            reward_survival = -0.1 * penalty_ratio
        else:
            reward_survival = 0.0

        # 4) 위험 페널티 (연속적 지수형)
        min_distance = np.min(lidar_scan)
        WARNING_DISTANCE = 0.25
        PENALTY_SCALE_DANGER = 8.0
        EXP_STEEPNESS = 10.0

        proximity = np.clip(WARNING_DISTANCE - min_distance, 0.0, None)
        danger_penalty = -PENALTY_SCALE_DANGER * (np.exp(EXP_STEEPNESS * proximity) - 1.0)
        danger_penalty = np.clip(danger_penalty, -20.0, 0.0)

        reward = reward_forward + reward_speed + reward_survival + danger_penalty

        # --- 종료 조건 ---

        # 1) 충돌 감지 (F1TENTH Gym 기본 플래그 사용)
        if raw_observation['collisions'][0]:
            terminated = True
            info['terminated'] = True
            info.setdefault('episode_ended_by', 'collision')
            collision_penalty = -10.0
            reward += collision_penalty
            if self.debug_mode:
                print(f"[에피소드 종료] 충돌 at step {self.episode_step}")

        # 2) 멈춤 감지 (stuck_check_interval 스텝마다)
        stuck_detected = False
        stuck_reason = None

        if (self.command_slow_steps >= self.command_slow_limit) and (self.episode_step > self.stuck_initial_delay):
            stuck_detected = True
            stuck_reason = 'stuck_command'

        if self.steps_since_last_check >= self.stuck_check_interval:
            if self.episode_step > self.stuck_initial_delay:
                movement = np.linalg.norm(pos_xy - self.last_check_pos)
                if (movement < self.stuck_threshold) and (command_speed > self.command_slow_threshold):
                    stuck_detected = True
                    stuck_reason = stuck_reason or 'stuck_movement'

            self.last_check_pos = pos_xy.copy()
            self.steps_since_last_check = 0

        if stuck_detected:
            terminated = True
            info['terminated'] = True
            info.setdefault('episode_ended_by', stuck_reason)
            if self.debug_mode:
                print(f"[Episode End] {stuck_reason.upper()} at step {self.episode_step}")
            self.command_slow_steps = 0

        # --- 에피소드 통계 ---
        self.episode_reward_sum += reward
        self.episode_forward_sum += reward_forward
        self.episode_speed_sum += reward_speed
        self.episode_survival_sum += reward_survival
        self.episode_danger_sum += danger_penalty
        self.episode_base_sum += base_reward

        episode_done = bool(terminated or truncated)

        # 에피소드 종료 시 디버그 로깅
        if episode_done and self.debug_mode:
            avg_reward = self.episode_reward_sum / max(self.episode_step, 1)
            print(f"[Episode Summary] Steps: {self.episode_step}, Total Distance: {self.total_distance:.2f}m")
            end_reason = info.get('episode_ended_by')
            if end_reason is None:
                if truncated and not terminated:
                    end_reason = 'timeout'
                elif terminated:
                    end_reason = 'unknown'
                else:
                    end_reason = 'none'

            print(f"  Rewards: Total={self.episode_reward_sum:.2f}, Avg={avg_reward:.3f}, Reason: {end_reason}")
            print(f"  Forward: {self.episode_forward_sum:.2f}, Speed: {self.episode_speed_sum:.2f}, "
                  f"Survival: {self.episode_survival_sum:.2f}, Danger: {self.episode_danger_sum:.2f}, "
                  f"Base: {self.episode_base_sum:.2f}")

        # 이전 위치 업데이트
        self.previous_pos = pos_xy.copy()

        info.setdefault('reward_components', {})
        info['reward_components'].update({
            'forward': reward_forward,
            'speed': reward_speed,
            'survival': reward_survival,
            'danger': danger_penalty,
            'base': base_reward,
        })

        terminated = bool(terminated)
        truncated = bool(truncated)
        info['terminated'] = terminated
        info['truncated'] = truncated

        return lidar_scan.astype(np.float32), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        환경 및 내부 상태를 리셋합니다.
        Centerline의 시작점에 차량을 스폰합니다.
        """
        # Centerline의 첫 점에서 시작 (일관된 학습을 위해)
        start_idx = 0
        start_xy = self.centerline_waypoints[start_idx]
        next_xy = self.centerline_waypoints[start_idx + 1]

        # 진행 방향 계산 (다음 웨이포인트를 향하도록)
        direction = np.arctan2(next_xy[1] - start_xy[1], next_xy[0] - start_xy[0])

        # 환경 리셋 (centerline 위치와 방향으로)
        lidar_scan, info = self.env.reset(start_xy=start_xy, direction=direction)
        info = {} if info is None else dict(info)
        raw_observation = info.get('raw_observation')
        if raw_observation is None:
            raise KeyError("raw_observation missing from inner env info during reset")

        # 진행 상황 추적 리셋
        pos_xy = np.array([raw_observation['poses_x'][0], raw_observation['poses_y'][0]])

        # 초기 진행 상황 찾기
        _, _ = self._get_track_progress(pos_xy)
        self.previous_pos = pos_xy.copy()
        self.lap_count = 0

        # 멈춤 감지 리셋
        self.last_check_pos = pos_xy.copy()
        self.steps_since_last_check = -self.stuck_initial_delay  # 첫 확인 지연

        # 에피소드 통계 리셋
        self.episode_step = 0
        self.episode_reward_sum = 0.0
        self.episode_forward_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_survival_sum = 0.0
        self.episode_danger_sum = 0.0
        self.episode_base_sum = 0.0

        info.setdefault('reward_components', {})

        return lidar_scan.astype(np.float32), info

    def seed(self, seed):
        self.env.seed(seed)
