#!/usr/bin/env python3
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


def transform_point_map_from_base(x_b, y_b, tf):
    # tf: geometry_msgs/TransformStamped (map->base_link)
    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    q = tf.transform.rotation
    # yaw from quaternion
    # Small helper (tf_transformations not imported here for yaw to save deps)
    # Compute rotation matrix elements for yaw-only extraction
    import numpy as np
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    # yaw
    yaw = math.atan2(2.0*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    # rotate and translate
    x_m = tx + (math.cos(yaw)*x_b - math.sin(yaw)*y_b)
    y_m = ty + (math.sin(yaw)*x_b + math.cos(yaw)*y_b)
    return x_m, y_m, yaw


class LocalAvoidanceNode(Node):
    def __init__(self):
        super().__init__('local_avoidance_node')

        # Parameters
        self.declare_parameter('global_path_topic', '/global_path')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('local_horizon', 8.0)  # meters
        self.declare_parameter('path_resolution', 0.2)  # meters
        self.declare_parameter('lateral_offsets', [0.0, 0.4, -0.4])  # meters
        self.declare_parameter('lookahead_distance', 2.0)  # meters
        self.declare_parameter('safety_radius', 0.4)  # meters
        self.declare_parameter('output_topic', '/local_path')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('map_frame', 'map')

        # Read params
        self.global_path_topic = self.get_parameter('global_path_topic').get_parameter_value().string_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.local_horizon = float(self.get_parameter('local_horizon').value)
        self.ds = float(self.get_parameter('path_resolution').value)
        self.lateral_offsets = [float(v) for v in self.get_parameter('lateral_offsets').value]
        self.safety_radius = float(self.get_parameter('safety_radius').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.global_path: Path = None
        self.sub_path = self.create_subscription(Path, self.global_path_topic, self._path_cb, 10)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, qos)

        # Publisher
        self.pub = self.create_publisher(Path, self.output_topic, 10)

        # State
        self.latest_scan: LaserScan = None

        # Timer to publish at fixed rate if data is available
        self.timer = self.create_timer(0.05, self._on_timer)  # 20 Hz

    def _path_cb(self, msg: Path):
        self.global_path = msg

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _get_tf(self):
        try:
            return self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def _closest_path_index(self, x, y) -> int:
        if self.global_path is None or not self.global_path.poses:
            return -1
        best_i = -1
        best_d2 = 1e18
        for i, ps in enumerate(self.global_path.poses):
            dx = ps.pose.position.x - x
            dy = ps.pose.position.y - y
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i
    
    def _get_global_path_points(self, current_x, current_y, current_yaw) -> List[Tuple[float, float]]:
        """Extract global path points within lookahead distance in robot's forward direction"""
        if self.global_path is None or not self.global_path.poses:
            return []
        
        # Find closest point on global path
        closest_idx = self._closest_path_index(current_x, current_y)
        if closest_idx < 0:
            return []
        
        # Determine forward direction along global path
        forward_idx, forward_direction = self._find_forward_direction(closest_idx, current_x, current_y, current_yaw)
        
        global_points = []
        cumulative_dist = 0.0
        prev_x, prev_y = current_x, current_y
        
        # Extract points along global path in forward direction within horizon
        if forward_direction > 0:
            # Forward direction (increasing index)
            for i in range(forward_idx, len(self.global_path.poses)):
                pose = self.global_path.poses[i]
                x = pose.pose.position.x
                y = pose.pose.position.y
                
                dist = math.hypot(x - prev_x, y - prev_y)
                cumulative_dist += dist
                
                if cumulative_dist > self.local_horizon:
                    break
                    
                global_points.append((x, y))
                prev_x, prev_y = x, y
        else:
            # Backward direction (decreasing index)
            for i in range(forward_idx, -1, -1):
                pose = self.global_path.poses[i]
                x = pose.pose.position.x
                y = pose.pose.position.y
                
                dist = math.hypot(x - prev_x, y - prev_y)
                cumulative_dist += dist
                
                if cumulative_dist > self.local_horizon:
                    break
                    
                global_points.append((x, y))
                prev_x, prev_y = x, y
            
        return global_points
    
    def _find_forward_direction(self, closest_idx, current_x, current_y, current_yaw) -> Tuple[int, int]:
        """Find which direction along global path aligns with robot's heading"""
        if not self.global_path or not self.global_path.poses:
            return closest_idx, 1
        
        # Get robot's forward direction vector
        robot_forward_x = math.cos(current_yaw)
        robot_forward_y = math.sin(current_yaw)
        
        best_idx = closest_idx
        best_direction = 1
        best_alignment = -2.0  # Initialize with worst possible dot product
        
        # Check forward direction (increasing index)
        if closest_idx + 1 < len(self.global_path.poses):
            next_pose = self.global_path.poses[closest_idx + 1]
            path_vec_x = next_pose.pose.position.x - current_x
            path_vec_y = next_pose.pose.position.y - current_y
            path_len = math.hypot(path_vec_x, path_vec_y)
            
            if path_len > 1e-6:
                path_vec_x /= path_len
                path_vec_y /= path_len
                alignment = robot_forward_x * path_vec_x + robot_forward_y * path_vec_y
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_idx = closest_idx
                    best_direction = 1
        
        # Check backward direction (decreasing index)
        if closest_idx - 1 >= 0:
            prev_pose = self.global_path.poses[closest_idx - 1]
            path_vec_x = prev_pose.pose.position.x - current_x
            path_vec_y = prev_pose.pose.position.y - current_y
            path_len = math.hypot(path_vec_x, path_vec_y)
            
            if path_len > 1e-6:
                path_vec_x /= path_len
                path_vec_y /= path_len
                alignment = robot_forward_x * path_vec_x + robot_forward_y * path_vec_y
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_idx = closest_idx
                    best_direction = -1
        
        return best_idx, best_direction
    
    def _transform_point_base_from_map(self, x_m, y_m, tf) -> Tuple[float, float]:
        """Transform point from map frame to base_link frame"""
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        yaw = math.atan2(2.0*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        # Translate to vehicle origin
        dx = x_m - tx
        dy = y_m - ty
        
        # Rotate to base_link frame
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        x_b = cos_yaw * dx - sin_yaw * dy
        y_b = sin_yaw * dx + cos_yaw * dy
        
        return x_b, y_b
    
    def _has_obstacle_in_direction(self, x_b, y_b, scan: LaserScan) -> bool:
        """Check if there's an obstacle in the direction of the target point"""
        if abs(x_b) < 1e-3 and abs(y_b) < 1e-3:
            return False
        
        theta = math.atan2(y_b, x_b)
        if theta < scan.angle_min or theta > scan.angle_max:
            return False
        
        idx = int(round((theta - scan.angle_min) / scan.angle_increment))
        if idx < 0 or idx >= len(scan.ranges):
            return False
        
        rng = scan.ranges[idx]
        if math.isnan(rng) or math.isinf(rng):
            return False
        
        target_dist = math.hypot(x_b, y_b)
        return rng < (target_dist + self.safety_radius)

    def _is_point_safe(self, x_b, y_b, scan: LaserScan) -> bool:
        r = math.hypot(x_b, y_b)
        if r < 1e-3:
            return True
        theta = math.atan2(y_b, x_b)
        # clamp to scan angle range
        if theta < scan.angle_min or theta > scan.angle_max:
            return True  # outside FOV, treat as safe to avoid over-constraining
        idx = int(round((theta - scan.angle_min) / scan.angle_increment))
        if idx < 0 or idx >= len(scan.ranges):
            return True
        rng = scan.ranges[idx]
        if math.isnan(rng) or math.isinf(rng):
            return True
        return (rng - r) > self.safety_radius

    def _build_local_path(self, tf_map_base, scan: LaserScan, current_x, current_y) -> List[Tuple[float, float]]:
        # Extract current robot yaw from transform
        q = tf_map_base.transform.rotation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        current_yaw = math.atan2(2.0*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        # Get global path points to follow in robot's forward direction
        global_points = self._get_global_path_points(current_x, current_y, current_yaw)
        
        if not global_points:
            # Fallback: generate straight path if no global path available
            return self._build_fallback_path(tf_map_base, scan)
        
        local_pts = []  # in map frame
        
        # Follow global path points, avoiding obstacles when necessary
        for global_point in global_points:
            gx, gy = global_point
            
            # Transform global point to base_link frame for obstacle checking
            gx_b, gy_b = self._transform_point_base_from_map(gx, gy, tf_map_base)
            
            # Check if direct path to global point is safe
            if not self._has_obstacle_in_direction(gx_b, gy_b, scan):
                # Safe to follow global path
                local_pts.append((gx, gy))
            else:
                # Obstacle detected, try lateral offsets from global point
                best_point = None
                
                # Try lateral offsets around the global point
                for offset in self.lateral_offsets[1:]:  # Skip center (already checked)
                    # Calculate offset point perpendicular to path direction
                    if len(local_pts) > 0:
                        # Use direction from last local point to global point
                        dx = gx - local_pts[-1][0]
                        dy = gy - local_pts[-1][1]
                    else:
                        # Use direction from current position to global point
                        dx = gx - current_x
                        dy = gy - current_y
                    
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        continue
                        
                    # Normalize direction vector
                    norm = math.hypot(dx, dy)
                    dx_norm = dx / norm
                    dy_norm = dy / norm
                    
                    # Calculate perpendicular offset point
                    offset_x = gx - offset * dy_norm  # perpendicular direction
                    offset_y = gy + offset * dx_norm
                    
                    # Check if offset point is safe
                    offset_x_b, offset_y_b = self._transform_point_base_from_map(offset_x, offset_y, tf_map_base)
                    
                    if not self._has_obstacle_in_direction(offset_x_b, offset_y_b, scan):
                        best_point = (offset_x, offset_y)
                        break
                
                if best_point:
                    local_pts.append(best_point)
                else:
                    # No safe path found, stop planning here
                    break
                    
            # Limit path length
            if len(local_pts) >= int(self.local_horizon / self.ds):
                break
        
        return local_pts
    
    def _build_fallback_path(self, tf_map_base, scan: LaserScan) -> List[Tuple[float, float]]:
        """Fallback path generation when global path is not available"""
        local_pts = []  # in base_link frame
        steps = max(2, int(self.local_horizon / self.ds))
        
        for k in range(steps):
            x = k * self.ds
            # choose lateral offset with best clearance
            best_off = None
            for off in self.lateral_offsets:
                if self._is_point_safe(x, off, scan):
                    best_off = off
                    break
            if best_off is None:
                # if none safe, shorten horizon
                break
            local_pts.append((x, best_off))
        
        # transform to map frame
        out = []
        for (xb, yb) in local_pts:
            xm, ym, _ = transform_point_map_from_base(xb, yb, tf_map_base)
            out.append((xm, ym))
        return out

    def _on_timer(self):
        if self.global_path is None or self.latest_scan is None:
            return
        tfmb = self._get_tf()
        if tfmb is None:
            return

        # Find closest point to current pose to stamp and orient local path roughly along global path
        cx = tfmb.transform.translation.x
        cy = tfmb.transform.translation.y
        idx = self._closest_path_index(cx, cy)
        if idx < 0:
            return

        # Build and publish Path
        pts_map = self._build_local_path(tfmb, self.latest_scan, cx, cy)
        if not pts_map:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        # Estimate yaw from first to last point for pose orientation
        yaw = 0.0
        if len(pts_map) >= 2:
            dx = pts_map[-1][0] - pts_map[0][0]
            dy = pts_map[-1][1] - pts_map[0][1]
            yaw = math.atan2(dy, dx)
        qz = math.sin(yaw/2.0)
        qw = math.cos(yaw/2.0)
        for (x, y) in pts_map:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self.pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = LocalAvoidanceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
