#!/usr/bin/env python3

"""
ROS2 노드: 학습된 SAC + CNN 정책을 사용하여 F1TENTH 차량을 제어합니다.

- /scan (sensor_msgs/LaserScan) 구독
- Stable Baselines 3 SAC 모델을 로드하여 [-1, 1] 범위의 행동을 예측
- 훈련 시 사용한 범위로 조향/속도를 역정규화하여 /drive (AckermannDriveStamped) 퍼블리시
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

from stable_baselines3 import SAC
import torch


def convert_range(value: float, input_range, output_range) -> float:
    """값을 다른 범위로 선형 변환합니다."""
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_span = in_max - in_min
    out_span = out_max - out_min
    if in_span == 0:
        return out_min
    return ((value - in_min) * out_span) / in_span + out_min


class SacCnnController(Node):
    """Stable Baselines 3 SAC 정책 기반 ROS 제어 노드."""

    def __init__(self) -> None:
        super().__init__('sac_cnn_controller')

        # --- 파라미터 선언 ---
        self.declare_parameter('model_path', str(Path.cwd() / 'train_sac_cnn' / 'sac-cnn-latest.zip'))
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('deterministic', True)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('drive_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('marker_topic', '/sac_control/marker')
        self.declare_parameter('filtered_scan_topic', '/filtered_scan')
        self.declare_parameter('lidar_clip', 30.0)
        self.declare_parameter('observation_dim', 1080)
        self.declare_parameter('replace_nan_with', 30.0)
        self.declare_parameter('replace_inf_with', 30.0)
        # 훈련 시와 동일한 action 범위
        # ⚠️ 기본값: --speed-cap 5.0으로 학습한 모델 기준
        # 다른 설정으로 학습했다면 launch 파라미터로 오버라이드
        self.declare_parameter('steering_min', -1.066)  # rad
        self.declare_parameter('steering_max', 1.066)   # rad
        self.declare_parameter('speed_min', 0.0)        # m/s
        self.declare_parameter('speed_max', 5.0)        # m/s
        # LiDAR 정규화 파라미터 (훈련 시 LidarNormalizeWrapper와 동일)
        self.declare_parameter('normalize_lidar', True)
        self.declare_parameter('lidar_mean', 15.0)      # 고정된 평균값 (근사)
        self.declare_parameter('lidar_std', 10.0)       # 고정된 표준편차 (근사)
        self.declare_parameter('normalize_clip', 5.0)   # 정규화 후 클리핑 범위

        # --- 파라미터 값 가져오기 ---
        self.model_path = Path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.drive_frame_id = self.get_parameter('drive_frame_id').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.filtered_scan_topic = self.get_parameter('filtered_scan_topic').get_parameter_value().string_value
        self.lidar_clip = float(self.get_parameter('lidar_clip').value)
        self.observation_dim = int(self.get_parameter('observation_dim').value)
        self.replace_nan_with = float(self.get_parameter('replace_nan_with').value)
        self.replace_inf_with = float(self.get_parameter('replace_inf_with').value)
        self.steering_min = float(self.get_parameter('steering_min').value)
        self.steering_max = float(self.get_parameter('steering_max').value)
        self.speed_min = float(self.get_parameter('speed_min').value)
        self.speed_max = float(self.get_parameter('speed_max').value)
        self.normalize_lidar = bool(self.get_parameter('normalize_lidar').value)
        self.lidar_mean = float(self.get_parameter('lidar_mean').value)
        self.lidar_std = float(self.get_parameter('lidar_std').value)
        self.normalize_clip = float(self.get_parameter('normalize_clip').value)

        # --- 모델 로드 ---
        self.model: Optional[SAC] = None
        if not self.model_path.exists():
            self.get_logger().error(f'SAC 모델 파일을 찾을 수 없습니다: {self.model_path}')
        else:
            try:
                # custom_objects를 사용하여 optimizer 클래스를 None으로 대체
                # 이렇게 하면 optimizer가 없어도 로드 가능
                import torch.optim as optim
                custom_objects = {
                    "optimizer_class": None,
                    "optimizer_kwargs": None,
                }
                self.model = SAC.load(self.model_path, device=self.device, custom_objects=custom_objects)
                self.get_logger().info(f'SAC 모델 로드 완료: {self.model_path}')
            except (KeyError, AttributeError) as exc:
                # Optimizer 로드 에러는 무시하고 policy만 로드
                if 'optimizer' in str(exc).lower() or 'policy' in str(exc).lower():
                    self.get_logger().warn(f'표준 로드 실패, policy만 로드 시도: {exc}')
                    try:
                        from stable_baselines3.common.save_util import load_from_zip_file
                        import torch

                        # Policy 가중치만 로드
                        data, params, pytorch_variables = load_from_zip_file(
                            self.model_path,
                            device=self.device,
                            print_system_info=False
                        )

                        # observation과 action space가 data에 있는지 확인
                        if 'observation_space' not in data or 'action_space' not in data:
                            raise ValueError("모델 파일에 observation_space 또는 action_space 정보가 없습니다.")

                        # 임시 SAC 모델 생성 (추론 전용)
                        from stable_baselines3.common.policies import ActorCriticPolicy
                        import gymnasium as gym

                        # 더미 환경 없이 policy만 생성
                        policy_class = data.get('policy_class', 'MlpPolicy')

                        # 간단한 방법: 저장된 policy state_dict만 로드
                        self.model = type('InferenceModel', (), {
                            'policy': None,
                            'predict': lambda self, obs, deterministic=True: self._predict(obs, deterministic),
                            '_predict': lambda self, obs, deterministic: self.policy.predict(obs, deterministic=deterministic)
                        })()

                        # Policy 객체 재구성
                        from code.cnn_policy import CNNSACPolicy

                        # Policy 생성
                        policy = CNNSACPolicy(
                            observation_space=data['observation_space'],
                            action_space=data['action_space'],
                            lr_schedule=lambda _: 0.0,  # 추론에는 사용 안 함
                        )

                        # 가중치 로드
                        policy.load_state_dict(params['policy'])
                        policy.to(self.device)
                        policy.eval()

                        self.model.policy = policy

                        self.get_logger().info(f'SAC 정책만 로드 완료 (추론 모드): {self.model_path}')
                    except Exception as inner_exc:
                        self.get_logger().error(f'SAC 정책 로드 실패: {inner_exc}')
                        import traceback
                        self.get_logger().error(traceback.format_exc())
                        self.model = None
                else:
                    self.get_logger().error(f'SAC 모델 로드 실패: {exc}')
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f'SAC 모델 로드 실패: {exc}')
                import traceback
                self.get_logger().error(traceback.format_exc())

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        self.filtered_scan_pub = self.create_publisher(LaserScan, self.filtered_scan_topic, 10)

        self.get_logger().info(f'/scan → {self.scan_topic}, /drive → {self.drive_topic}')
        self.get_logger().info(f'CNN features → {self.filtered_scan_topic}')
        self.get_logger().info(f'조향 범위: [{self.steering_min:.3f}, {self.steering_max:.3f}] rad, '
                              f'속도 범위: [{self.speed_min:.1f}, {self.speed_max:.1f}] m/s')
        if self.normalize_lidar:
            self.get_logger().info(f'LiDAR 정규화: ON (mean={self.lidar_mean:.1f}, std={self.lidar_std:.1f}, clip=±{self.normalize_clip:.1f})')
        else:
            self.get_logger().info(f'LiDAR 정규화: OFF (원시 값 사용)')

    def preprocess_scan(self, msg: LaserScan) -> Optional[np.ndarray]:
        """
        LiDAR 데이터를 SAC 정책 입력 형식으로 변환.
        훈련 시와 동일한 전처리 적용:
        1. NaN/Inf 처리
        2. Clipping
        3. 정규화 (옵션)
        """
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size != self.observation_dim:
            self.get_logger().warn(
                f'LiDAR 포인트 수({ranges.size})가 기대값({self.observation_dim})과 다릅니다. 메시지 무시.',
                throttle_duration_sec=1.0
            )
            return None

        # 1. NaN/Inf 처리
        ranges = np.nan_to_num(
            ranges,
            nan=self.replace_nan_with,
            posinf=self.replace_inf_with,
            neginf=0.0
        )

        # 2. Clipping (원시 값 범위)
        ranges = np.clip(ranges, 0.0, self.lidar_clip)

        # 3. 정규화 (훈련 시 LidarNormalizeWrapper와 동일)
        if self.normalize_lidar:
            # (observation - mean) / std
            normalized = (ranges - self.lidar_mean) / (self.lidar_std + 1e-6)
            # Clip to [-normalize_clip, +normalize_clip]
            normalized = np.clip(normalized, -self.normalize_clip, self.normalize_clip)
            return normalized.astype(np.float32)

        return ranges.astype(np.float32)

    def scan_callback(self, msg: LaserScan) -> None:
        """LiDAR 콜백에서 SAC 모델을 사용하여 AckermannDrive 명령을 발행."""
        if self.model is None:
            return

        observation = self.preprocess_scan(msg)
        if observation is None:
            return

        try:
            # CNN 특징 추출 + 원본 LiDAR 전달
            cnn_features = self.extract_cnn_features(observation)
            if cnn_features is not None:
                # 원본 LiDAR도 함께 전달
                self.publish_filtered_scan(cnn_features, msg.ranges)

            # 행동 예측
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'SAC 예측 실패: {exc}')
            return

        steer = convert_range(float(action[0]), (-1.0, 1.0), (self.steering_min, self.steering_max))
        speed = convert_range(float(action[1]), (-1.0, 1.0), (self.speed_min, self.speed_max))

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = self.drive_frame_id
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)
        self.publish_marker(steer, speed, drive_msg.header.stamp)

    def extract_cnn_features(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """
        LiDAR 관측값에서 CNN 거리 다운샘플러 출력을 추출합니다.

        새 모델: 1080 → CNN → 128개 다운샘플링된 거리값
        구 모델: 1080 → CNN → 64개 추상 특징

        Args:
            observation: 전처리된 LiDAR 스캔 (1080,)

        Returns:
            128차원 거리값 또는 64차원 추상 특징, 실패 시 None
        """
        try:
            # Policy 접근
            if not hasattr(self.model, 'policy') or self.model.policy is None:
                self.get_logger().warn('Model policy not found', throttle_duration_sec=10.0)
                return None

            policy = self.model.policy

            # features_extractor 찾기 (SAC는 actor 안에 있음)
            features_extractor = None

            # 1. Actor의 features_extractor 확인 (SAC의 경우)
            if hasattr(policy, 'actor') and policy.actor is not None:
                if hasattr(policy.actor, 'features_extractor') and policy.actor.features_extractor is not None:
                    features_extractor = policy.actor.features_extractor
                    self.get_logger().info('Using actor.features_extractor', once=True)

            # 2. Policy의 features_extractor 확인 (fallback)
            if features_extractor is None:
                if hasattr(policy, 'features_extractor') and policy.features_extractor is not None:
                    features_extractor = policy.features_extractor
                    self.get_logger().info('Using policy.features_extractor', once=True)

            if features_extractor is None:
                self.get_logger().warn('Features extractor not available in actor or policy', throttle_duration_sec=10.0)
                return None

            # Observation을 torch tensor로 변환
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Device를 torch.device로 변환
            device = torch.device(self.device)
            obs_tensor = obs_tensor.to(device)

            # Features extractor를 evaluation 모드로 설정
            features_extractor.eval()

            # Features 추출 (자동으로 거리 다운샘플러 또는 추상 특징 반환)
            with torch.no_grad():
                features = features_extractor(obs_tensor)  # (batch, 128) 또는 (batch, 64)
                features_np = features.cpu().numpy().flatten()

                # 모델 타입 로깅 (한 번만)
                if features_np.shape[0] == 128:
                    self.get_logger().info('거리 다운샘플러 모델 (128개 거리값)', once=True)
                else:
                    self.get_logger().info(f'추상 특징 모델 ({features_np.shape[0]}개 특징)', once=True)

                return features_np

        except Exception as exc:
            self.get_logger().warn(f'CNN 특징 추출 실패: {exc}', throttle_duration_sec=5.0)
            return None

    def publish_filtered_scan(self, cnn_features: np.ndarray, original_ranges: list) -> None:
        """
        CNN 출력을 LaserScan으로 발행합니다.

        두 가지 모드:
        1. 거리 다운샘플러 모드 (128개): CNN 출력을 직접 거리로 사용
        2. 추상 특징 모드 (기타): 원본 다운샘플링 + CNN intensity

        Args:
            cnn_features: CNN 출력 (128개 거리값 또는 다른 크기 특징)
            original_ranges: 원본 LiDAR 거리값 (1080,)
        """
        if self.filtered_scan_pub is None:
            return

        # LaserScan 메시지 생성
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/laser'

        num_features = len(cnn_features)
        fov = 4.7  # 270도
        msg.angle_min = -fov / 2.0
        msg.angle_max = fov / 2.0
        msg.angle_increment = fov / (num_features - 1) if num_features > 1 else 0.0
        msg.time_increment = 0.0
        msg.scan_time = 0.025  # 40Hz
        msg.range_min = 0.0
        msg.range_max = 30.0

        if num_features == 128:
            # ===== 거리 다운샘플러 모드 =====
            # CNN 출력이 이미 거리값! (1080 → 128 학습된 다운샘플링)
            msg.ranges = cnn_features.tolist()

            # Intensity는 원본 LiDAR 다운샘플링 (비교용)
            original_ranges_arr = np.array(original_ranges)
            downsample_factor = len(original_ranges_arr) // num_features  # 1080 // 128 ≈ 8.4

            original_downsampled = []
            for i in range(num_features):
                start_idx = int(i * downsample_factor)
                end_idx = int((i + 1) * downsample_factor)
                range_segment = original_ranges_arr[start_idx:end_idx]
                original_downsampled.append(float(np.min(range_segment)))

            msg.intensities = original_downsampled

        else:
            # ===== 추상 특징 모드 (기존 방식) =====
            # 원본 LiDAR 다운샘플링
            original_ranges_arr = np.array(original_ranges)
            downsample_factor = len(original_ranges_arr) // num_features

            downsampled_ranges = []
            for i in range(num_features):
                start_idx = i * downsample_factor
                end_idx = start_idx + downsample_factor
                range_segment = original_ranges_arr[start_idx:end_idx]
                downsampled_ranges.append(float(np.min(range_segment)))

            msg.ranges = downsampled_ranges

            # CNN 활성화를 intensity로
            cnn_intensities = np.maximum(cnn_features, 0.0)
            if cnn_intensities.max() > 0.01:
                normalized_intensities = (cnn_intensities / cnn_intensities.max()) * 1000.0
            else:
                normalized_intensities = np.zeros_like(cnn_intensities)

            msg.intensities = normalized_intensities.tolist()

        self.filtered_scan_pub.publish(msg)

    def publish_marker(self, steer: float, speed: float, stamp) -> None:
        """조향/속도 값을 RViz에서 확인할 수 있도록 Marker를 출력."""
        if self.marker_pub is None:
            return

        marker = Marker()
        marker.header.frame_id = self.drive_frame_id
        marker.header.stamp = stamp
        marker.ns = 'sac_cnn_control'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.1

        yaw = steer
        marker.pose.orientation.z = math.sin(yaw / 2.0)
        marker.pose.orientation.w = math.cos(yaw / 2.0)

        marker.scale.x = max(abs(speed), 0.1)
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

        text_marker = Marker()
        text_marker.header.frame_id = self.drive_frame_id
        text_marker.header.stamp = stamp
        text_marker.ns = 'sac_cnn_control'
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = 0.6
        text_marker.scale.z = 0.25
        text_marker.color.r = 0.9
        text_marker.color.g = 0.9
        text_marker.color.b = 0.9
        text_marker.color.a = 1.0
        text_marker.text = f"speed: {speed:.2f} m/s\nsteer: {steer:.3f} rad"

        self.marker_pub.publish(text_marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SacCnnController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SAC CNN Controller 종료 요청')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
