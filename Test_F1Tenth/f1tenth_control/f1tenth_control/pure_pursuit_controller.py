#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener


class PurePursuitController(Node):
    """
    Pure Pursuit Controller for F1TENTH
    - Subscribes to global/local path and odometry
    - Publishes Ackermann drive commands
    - Uses local path (priority) and global path (fallback) for steering control
    - Fixed speed control with pure pursuit steering algorithm
    """

    def __init__(self):
        super().__init__('pure_pursuit_controller')
        
        # Parameters
        self.declare_parameter('lookahead_distance', 2.5)  # meters (increased for better cornering)
        self.declare_parameter('speed', 0.3)  # m/s (fixed speed)
        self.declare_parameter('wheelbase', 0.3302)  # meters (F1TENTH wheelbase)
        self.declare_parameter('max_steering_angle', 0.4189)  # radians (~24 degrees)
        self.declare_parameter('path_topic', '/local_path')  # prefer local over global
        self.declare_parameter('fallback_path_topic', '/global_path')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')  # f1tenth_gym_ros namespace
        self.declare_parameter('drive_topic', '/drive')

        # Get parameters
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.speed = self.get_parameter('speed').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        
        # State variables
        self.current_pose = None
        self.global_path = None
        self.local_path = None
        self.local_path_timestamp = None
        self.local_path_timeout = 1.0  # seconds
        
        # TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # QoS profiles
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            qos
        )
        
        self.path_sub = self.create_subscription(
            Path,
            self.get_parameter('path_topic').value,
            self.path_callback,
            10
        )
        
        self.fallback_path_sub = self.create_subscription(
            Path,
            self.get_parameter('fallback_path_topic').value,
            self.fallback_path_callback,
            10
        )
        
        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('drive_topic').value,
            10
        )
        
        # Control timer (50 Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Pure Pursuit Controller initialized')
        self.get_logger().info(f'Lookahead distance: {self.lookahead_distance}m')
        self.get_logger().info(f'Fixed speed: {self.speed} m/s')
        self.get_logger().info('Local path priority enabled for steering control')

    def odom_callback(self, msg):
        """Update current vehicle pose from odometry"""
        self.current_pose = msg.pose.pose

    def path_callback(self, msg):
        """Local path callback"""
        if len(msg.poses) > 0:
            self.local_path = msg
            self.local_path_timestamp = self.get_clock().now()
            # self.get_logger().info(f'Received local path with {len(msg.poses)} waypoints')

    def fallback_path_callback(self, msg):
        """Global path callback"""
        if len(msg.poses) > 0:
            self.global_path = msg
            # self.get_logger().info(f'Received global path with {len(msg.poses)} waypoints')

    def get_current_path(self):
        """
        Get current path with priority: local path (if recent) > global path
        """
        current_time = self.get_clock().now()
        
        # Check if local path is recent and available
        if (self.local_path is not None and 
            self.local_path_timestamp is not None and 
            len(self.local_path.poses) > 0):
            
            time_diff = (current_time - self.local_path_timestamp).nanoseconds / 1e9
            if time_diff < self.local_path_timeout:
                # self.get_logger().debug('Using local path for steering control')
                return self.local_path
        
        # Use global path as fallback
        if self.global_path is not None and len(self.global_path.poses) > 0:
            # self.get_logger().debug('Using global path as fallback for steering control')
            return self.global_path
            
        return None

    def find_target_point(self):
        """
        Find target point on path using lookahead distance
        Returns target point coordinates (target_x, target_y) or None if not found
        """
        path = self.get_current_path()
        if not path or not self.current_pose:
            return None
            
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Get vehicle heading for forward-looking search
        q = self.current_pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        # Find closest point on path (forward-looking)
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose_stamped in enumerate(path.poses):
            dx = pose_stamped.pose.position.x - current_x
            dy = pose_stamped.pose.position.y - current_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Check if point is roughly in front of vehicle
            to_point_angle = math.atan2(dy, dx)
            angle_diff = abs(to_point_angle - yaw)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # wrap to [0, pi]
            
            # Prefer points that are in front (within 90 degrees)
            if angle_diff < math.pi/2 and dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Search forward from closest point for lookahead distance
        for i in range(closest_idx, len(path.poses)):
            target_x = path.poses[i].pose.position.x
            target_y = path.poses[i].pose.position.y
            
            dx = target_x - current_x
            dy = target_y - current_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist >= self.lookahead_distance:
                return target_x, target_y
        
        # If no point found at lookahead distance, use last point
        if len(path.poses) > 0:
            last_pose = path.poses[-1].pose
            return last_pose.position.x, last_pose.position.y
            
        return None

    def pure_pursuit_control(self, target_point):
        """
        Calculate steering angle using pure pursuit algorithm
        """
        if not target_point or not self.current_pose:
            return 0.0
            
        # Current vehicle state
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Get vehicle heading from quaternion
        q = self.current_pose.orientation
        # Convert quaternion to yaw
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        # Transform target point to vehicle frame
        target_x, target_y = target_point
        
        # Vector from vehicle to target
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Rotate to vehicle frame
        target_x_vehicle = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        target_y_vehicle = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        
        # Calculate lookahead distance (actual distance to target)
        lookahead_actual = math.sqrt(target_x_vehicle**2 + target_y_vehicle**2)
        
        if lookahead_actual < 0.1:  # Too close, no steering needed
            return 0.0
        
        # Pure pursuit steering calculation
        # steering_angle = atan2(2 * L * sin(alpha), lookahead_distance)
        # where alpha is the angle between vehicle heading and target direction
        alpha = math.atan2(target_y_vehicle, target_x_vehicle)
        
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), lookahead_actual)
        
        # Limit steering angle
        steering_angle = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, steering_angle))
        
        return steering_angle

    def control_loop(self):
        """Main control loop - runs at 50Hz"""
        current_path = self.get_current_path()
        if not self.current_pose or not current_path:
            # Stop if no path or pose
            self.publish_drive_command(0.0, 0.0)
            return
        
        # Find target point for steering control
        target_point = self.find_target_point()
        if not target_point:
            self.get_logger().warn('No target point found on path, stopping')
            self.publish_drive_command(0.0, 0.0)
            return
        
        # Calculate steering angle using pure pursuit
        steering_angle = self.pure_pursuit_control(target_point)
        
        # Publish drive command with fixed speed and calculated steering
        self.publish_drive_command(self.speed, steering_angle)

    def publish_drive_command(self, speed, steering_angle):
        """Publish Ackermann drive command"""
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Pure Pursuit Controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()