#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from builtin_interfaces.msg import Time
import csv


class GlobalCheckpointNode(Node):
    def __init__(self):
        super().__init__('global_checkpoint_node')

        # Parameters
        self.declare_parameter('checkpoint_csv_path', '')
        self.declare_parameter('publish_topic', '/global_path')
        self.declare_parameter('initial_pose_topic', '/initialpose')

        self.csv_path = self.get_parameter('checkpoint_csv_path').get_parameter_value().string_value
        self.topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        if not self.csv_path:
            raise RuntimeError('checkpoint_csv_path parameter is required')

        # Load checkpoints from CSV
        self.waypoints_xy = self._load_checkpoints(self.csv_path)
        self.get_logger().info(f'Loaded {len(self.waypoints_xy)} checkpoints from {self.csv_path}')

        # Starting waypoint index (will be updated by 2D Pose Estimate)
        self.start_index = 0

        # Prepare Path message
        self.frame_id = 'map'

        # Publisher (latched-like behavior via transient local)
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Path, self.topic, qos)

        # Subscribe to /initialpose (2D Pose Estimate from RViz)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self._initial_pose_callback,
            10
        )

        # Periodic publish to keep timestamps fresh
        self.timer = self.create_timer(1.0, self._publish_path)
        self._publish_path()

    def _load_checkpoints(self, csv_path):
        """Load checkpoints from CSV file (x, y columns)"""
        waypoints = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = float(row['x'])
                    y = float(row['y'])
                    waypoints.append((x, y))
                except (ValueError, KeyError):
                    # Skip invalid rows
                    pass
        return waypoints

    def _initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Callback for 2D Pose Estimate from RViz.
        Find the closest waypoint that matches the direction of the pose.
        """
        pose_x = msg.pose.pose.position.x
        pose_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        pose_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Direction vector from pose
        pose_dir = np.array([np.cos(pose_yaw), np.sin(pose_yaw)])

        # Find closest waypoint with matching direction
        best_index = 0
        best_score = -float('inf')

        for i in range(len(self.waypoints_xy)):
            wx, wy = self.waypoints_xy[i]

            # Distance to waypoint
            dist = np.sqrt((wx - pose_x)**2 + (wy - pose_y)**2)

            # Direction to next waypoint
            next_i = (i + 1) % len(self.waypoints_xy)
            next_wx, next_wy = self.waypoints_xy[next_i]
            wp_dir = np.array([next_wx - wx, next_wy - wy])
            wp_dir_norm = np.linalg.norm(wp_dir)

            if wp_dir_norm > 0.001:
                wp_dir = wp_dir / wp_dir_norm

                # Dot product: alignment with pose direction
                alignment = np.dot(pose_dir, wp_dir)

                # Score: prefer closer waypoints with better alignment
                # Higher alignment and lower distance give higher score
                score = alignment - 0.5 * dist

                if score > best_score:
                    best_score = score
                    best_index = i

        self.start_index = best_index
        self.get_logger().info(
            f'2D Pose Estimate received: ({pose_x:.2f}, {pose_y:.2f}, yaw={np.degrees(pose_yaw):.1f}°). '
            f'Starting from waypoint index {self.start_index}'
        )

        # Publish updated path immediately
        self._publish_path()

    def _stamp_now(self) -> Time:
        return self.get_clock().now().to_msg()

    def _publish_path(self):
        """Publish the path starting from start_index"""
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = self._stamp_now()

        # Create circular path starting from start_index
        n = len(self.waypoints_xy)
        for i in range(n):
            idx = (self.start_index + i) % n
            x, y = self.waypoints_xy[idx]

            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = path_msg.header.stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # Close the loop by adding the first waypoint again
        if len(self.waypoints_xy) > 0:
            x, y = self.waypoints_xy[self.start_index]
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = path_msg.header.stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCheckpointNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
