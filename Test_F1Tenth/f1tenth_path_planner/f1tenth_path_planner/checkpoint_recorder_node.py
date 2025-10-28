#!/usr/bin/env python3
import csv
import math
import os
from typing import List, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Compute yaw (rotation about Z) from quaternion components."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1 - 2.0 * (qy * qy + qz * qz))


class CheckpointRecorderNode(Node):
    """Node that records checkpoints from the current TF pose and saves them to CSV."""

    def __init__(self):
        super().__init__('checkpoint_recorder_node')

        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('output_csv_path', '')
        self.declare_parameter('auto_save_on_add', True)
        self.declare_parameter('publish_topic', '/checkpoint_path')
        self.declare_parameter('clicked_point_topic', '/clicked_point')

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        output_param = self.get_parameter('output_csv_path').get_parameter_value().string_value
        if output_param:
            csv_path = output_param
        else:
            share_dir = get_package_share_directory('f1tenth_path_planner')
            csv_path = os.path.join(share_dir, 'data', 'checkpoints.csv')
        self.output_csv_path = os.path.expanduser(csv_path)
        self.auto_save = bool(self.get_parameter('auto_save_on_add').value)
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.clicked_topic = self.get_parameter('clicked_point_topic').get_parameter_value().string_value

        self._checkpoints: List[Tuple[float, float]] = []

        # TF listener to retrieve current position on demand.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions / Services
        self.sub_clicked = self.create_subscription(PointStamped, self.clicked_topic, self._clicked_point_cb, 10)
        self.save_srv = self.create_service(Trigger, 'save_checkpoints', self._handle_save_checkpoints)
        self.clear_srv = self.create_service(Trigger, 'clear_checkpoints', self._handle_clear_checkpoints)

        # Publisher with transient local QoS so recorded path stays latched.
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, self.publish_topic, qos)
        self._publish_path()

        self.get_logger().info(
            f'Checkpoint recorder ready (map_frame={self.map_frame}, clicked_topic={self.clicked_topic}, '
            f'output="{self.output_csv_path}")'
        )

    def _transform_point_to_map(self, msg: PointStamped) -> Tuple[float, float]:
        """Transform incoming point into map frame if required."""
        if not msg.header.frame_id or msg.header.frame_id == self.map_frame:
            return float(msg.point.x), float(msg.point.y)

        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn(
                f'No transform from {msg.header.frame_id} to {self.map_frame}; skipping point: {exc}')
            return None

        trans = tf.transform.translation
        rot = tf.transform.rotation
        yaw = _yaw_from_quaternion(rot.x, rot.y, rot.z, rot.w)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = msg.point.x
        y = msg.point.y
        # Rotate then translate
        map_x = trans.x + cos_yaw * x - sin_yaw * y
        map_y = trans.y + sin_yaw * x + cos_yaw * y
        return float(map_x), float(map_y)

    def _clicked_point_cb(self, msg: PointStamped):
        result = self._transform_point_to_map(msg)
        if result is None:
            return

        self._checkpoints.append(result)
        self.get_logger().info(
            f'Checkpoint #{len(self._checkpoints)} recorded at x={result[0]:.2f}, y={result[1]:.2f}')
        self._publish_path()

        if self.auto_save:
            self._write_csv()

    def _handle_save_checkpoints(self, request, response):
        if not self._checkpoints:
            response.success = False
            response.message = 'No checkpoints to save.'
            return response

        try:
            path = self._write_csv()
        except Exception as exc:
            response.success = False
            response.message = f'Failed to save CSV: {exc}'
            return response

        response.success = True
        response.message = f'Saved {len(self._checkpoints)} checkpoints to {path}'
        return response

    def _handle_clear_checkpoints(self, request, response):
        count = len(self._checkpoints)
        self._checkpoints.clear()
        self._publish_path()
        # Rewrite CSV with header only.
        self._write_csv()
        response.success = True
        response.message = f'Cleared {count} checkpoints.'
        return response

    def _publish_path(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for idx, (x, y) in enumerate(self._checkpoints):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            yaw = self._estimate_yaw(idx)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _estimate_yaw(self, index: int) -> float:
        if len(self._checkpoints) <= 1:
            return 0.0
        if index < len(self._checkpoints) - 1:
            next_x, next_y = self._checkpoints[index + 1]
            curr_x, curr_y = self._checkpoints[index]
        else:
            curr_x, curr_y = self._checkpoints[index]
            prev_x, prev_y = self._checkpoints[index - 1]
            next_x, next_y = curr_x, curr_y
            curr_x, curr_y = prev_x, prev_y
        dx = next_x - curr_x
        dy = next_y - curr_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return math.atan2(dy, dx)

    def _write_csv(self) -> str:
        directory = os.path.dirname(self.output_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            writer.writerows(self._checkpoints)
        self.get_logger().info(f'Saved {len(self._checkpoints)} checkpoints to {self.output_csv_path}')
        return self.output_csv_path


def main(args=None):
    rclpy.init(args=args)
    node = CheckpointRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
