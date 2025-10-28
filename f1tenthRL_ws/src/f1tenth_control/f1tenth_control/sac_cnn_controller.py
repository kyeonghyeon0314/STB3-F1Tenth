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
        self.declare_parameter('drive_frame_id', 'base_link')
        self.declare_parameter('marker_topic', '/sac_control/marker')
        self.declare_parameter('lidar_clip', 30.0)
        self.declare_parameter('observation_dim', 1080)
        self.declare_parameter('replace_nan_with', 30.0)
        self.declare_parameter('replace_inf_with', 30.0)
        self.declare_parameter('steering_min', -0.4189)
        self.declare_parameter('steering_max', 0.4189)
        self.declare_parameter('speed_min', -5.0)
        self.declare_parameter('speed_max', 20.0)

        # --- 파라미터 값 가져오기 ---
        self.model_path = Path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.drive_frame_id = self.get_parameter('drive_frame_id').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.lidar_clip = float(self.get_parameter('lidar_clip').value)
        self.observation_dim = int(self.get_parameter('observation_dim').value)
        self.replace_nan_with = float(self.get_parameter('replace_nan_with').value)
        self.replace_inf_with = float(self.get_parameter('replace_inf_with').value)
        self.steering_min = float(self.get_parameter('steering_min').value)
        self.steering_max = float(self.get_parameter('steering_max').value)
        self.speed_min = float(self.get_parameter('speed_min').value)
        self.speed_max = float(self.get_parameter('speed_max').value)

        # --- 모델 로드 ---
        self.model: Optional[SAC] = None
        if not self.model_path.exists():
            self.get_logger().error(f'SAC 모델 파일을 찾을 수 없습니다: {self.model_path}')
        else:
            try:
                self.model = SAC.load(self.model_path, device=self.device)
                self.get_logger().info(f'SAC 모델 로드 완료: {self.model_path}')
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f'SAC 모델 로드 실패: {exc}')

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)

        self.get_logger().info(f'/scan → {self.scan_topic}, /drive → {self.drive_topic}')
        self.get_logger().info(f'조향 범위: [{self.steering_min}, {self.steering_max}], 속도 범위: [{self.speed_min}, {self.speed_max}]')

    def preprocess_scan(self, msg: LaserScan) -> Optional[np.ndarray]:
        """LiDAR 데이터를 SAC 정책 입력 형식으로 변환."""
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size != self.observation_dim:
            self.get_logger().warn(
                f'LiDAR 포인트 수({ranges.size})가 기대값({self.observation_dim})과 다릅니다. 메시지 무시.',
                throttle_duration_sec=1.0
            )
            return None

        ranges = np.nan_to_num(
            ranges,
            nan=self.replace_nan_with,
            posinf=self.replace_inf_with,
            neginf=0.0
        )
        ranges = np.clip(ranges, 0.0, self.lidar_clip)
        return ranges.astype(np.float32)

    def scan_callback(self, msg: LaserScan) -> None:
        """LiDAR 콜백에서 SAC 모델을 사용하여 AckermannDrive 명령을 발행."""
        if self.model is None:
            return

        observation = self.preprocess_scan(msg)
        if observation is None:
            return

        try:
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
