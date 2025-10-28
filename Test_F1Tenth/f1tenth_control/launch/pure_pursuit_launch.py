#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch arguments
    lookahead_distance_arg = DeclareLaunchArgument(
        'lookahead_distance',
        default_value='1.0',
        description='Lookahead distance for pure pursuit algorithm (meters)'
    )
    
    speed_arg = DeclareLaunchArgument(
        'speed',
        default_value='1.2',
        description='Fixed speed for the vehicle (m/s)'
    )
    
    wheelbase_arg = DeclareLaunchArgument(
        'wheelbase',
        default_value='0.3302',
        description='Vehicle wheelbase length (meters)'
    )
    
    max_steering_angle_arg = DeclareLaunchArgument(
        'max_steering_angle',
        default_value='2.4',
        description='Maximum steering angle (radians, ~24 degrees)'
    )
    
    path_topic_arg = DeclareLaunchArgument(
        'path_topic',
        default_value='/local_path',
        description='Primary path topic (local path preferred)'
    )
    
    fallback_path_topic_arg = DeclareLaunchArgument(
        'fallback_path_topic',
        default_value='/global_path',
        description='Fallback path topic (global path)'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/ego_racecar/odom',
        description='Odometry topic'
    )
    
    drive_topic_arg = DeclareLaunchArgument(
        'drive_topic',
        default_value='/drive',
        description='Ackermann drive command topic'
    )

    # Pure Pursuit Controller Node
    pure_pursuit_node = Node(
        package='f1tenth_control',
        executable='pure_pursuit_controller',
        name='pure_pursuit_controller',
        parameters=[{
            'lookahead_distance': LaunchConfiguration('lookahead_distance'),
            'speed': LaunchConfiguration('speed'),
            'wheelbase': LaunchConfiguration('wheelbase'),
            'max_steering_angle': LaunchConfiguration('max_steering_angle'),
            'path_topic': LaunchConfiguration('path_topic'),
            'fallback_path_topic': LaunchConfiguration('fallback_path_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'drive_topic': LaunchConfiguration('drive_topic'),
        }],
        output='screen'
    )

    return LaunchDescription([
        lookahead_distance_arg,
        speed_arg,
        wheelbase_arg,
        max_steering_angle_arg,
        path_topic_arg,
        fallback_path_topic_arg,
        odom_topic_arg,
        drive_topic_arg,
        pure_pursuit_node
    ])