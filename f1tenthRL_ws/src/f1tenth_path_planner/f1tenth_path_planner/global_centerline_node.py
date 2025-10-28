#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from ament_index_python.packages import get_package_share_directory
import yaml

from .utils import extract_centerline_image, order_skeleton_points, load_map_info, downsample_loop


class GlobalCenterlineNode(Node):
    def __init__(self):
        super().__init__('global_centerline_node')

        # Parameters
        self.declare_parameter('map_path', '')  # map root without extension, e.g., /.../maps/Spielberg_map
        self.declare_parameter('map_img_ext', '.png')
        self.declare_parameter('map_yaml_path', '')
        self.declare_parameter('sample_step_m', 0.2)
        self.declare_parameter('publish_topic', '/global_path')
        self.declare_parameter('save_centerline_overlay', True)

        map_path = self.get_parameter('map_path').get_parameter_value().string_value
        map_img_ext = self.get_parameter('map_img_ext').get_parameter_value().string_value
        map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value
        self.sample_step = float(self.get_parameter('sample_step_m').value)
        self.topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        save_overlay = bool(self.get_parameter('save_centerline_overlay').value)

        # If map_path not specified, try reading f1tenth_gym_ros config/sim.yaml
        if not map_path:
            try:
                fgr = get_package_share_directory('f1tenth_gym_ros')
                sim_yaml = os.path.join(fgr, 'config', 'sim.yaml')
                with open(sim_yaml, 'r') as f:
                    cfg = yaml.safe_load(f)
                map_path = cfg['bridge']['ros__parameters']['map_path']
                map_img_ext = cfg['bridge']['ros__parameters']['map_img_ext']
                self.get_logger().info(f'Loaded map_path from f1tenth_gym_ros: {map_path}{map_img_ext}')
            except Exception as e:
                self.get_logger().warn(f'Failed to read f1tenth_gym_ros sim.yaml: {e}')

        if not map_yaml_path and map_path:
            map_yaml_path = map_path + '.yaml'

        # Resolve paths
        if not map_path:
            raise RuntimeError('map_path parameter is required (map root path without extension).')
        img_path = map_path + map_img_ext
        yaml_path = map_yaml_path

        # Extract skeleton and waypoints
        centerline_overlay_path = map_path + '_centerline.png'
        skeleton = extract_centerline_image(img_path, centerline_overlay_path) if save_overlay else None
        if skeleton is None:
            # still need a skeleton mask for waypoints
            skeleton = extract_centerline_image(img_path, centerline_overlay_path)

        pts_uv = order_skeleton_points(skeleton)
        map_info = load_map_info(yaml_path)
        waypoints_xy = downsample_loop(pts_uv, map_info, step_meters=self.sample_step)
        self.get_logger().info(f'Centerline waypoints: {len(waypoints_xy)}')

        # Prepare Path message
        self.frame_id = 'map'
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id
        self.path_msg.poses = []
        for (x, y) in waypoints_xy:
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            self.path_msg.poses.append(ps)

        # Publisher (latched-like behavior via transient local)
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Path, self.topic, qos)

        # Periodic publish to keep timestamps fresh
        self.timer = self.create_timer(1.0, self._publish_once)
        self._publish_once()

    def _stamp_now(self) -> Time:
        return self.get_clock().now().to_msg()

    def _publish_once(self):
        ts = self._stamp_now()
        self.path_msg.header.stamp = ts
        for ps in self.path_msg.poses:
            ps.header.stamp = ts
        self.pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCenterlineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
