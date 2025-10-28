#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='train_sac_cnn/sac-cnn-latest.zip',
        description='Stable Baselines 3 SAC 모델 (.zip) 경로'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='모델 추론 장치(cpu/cuda)'
    )

    deterministic_arg = DeclareLaunchArgument(
        'deterministic',
        default_value='true',
        description='정책을 결정적으로 사용할지 여부'
    )

    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='LiDAR LaserScan 토픽'
    )

    drive_topic_arg = DeclareLaunchArgument(
        'drive_topic',
        default_value='/drive',
        description='AckermannDriveStamped 퍼블리시 토픽'
    )

    drive_frame_arg = DeclareLaunchArgument(
        'drive_frame_id',
        default_value='base_link',
        description='드라이브 메시지 frame_id'
    )

    sac_node = Node(
        package='f1tenth_control',
        executable='sac_cnn_controller',
        name='sac_cnn_controller',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'device': LaunchConfiguration('device'),
            'deterministic': LaunchConfiguration('deterministic'),
            'scan_topic': LaunchConfiguration('scan_topic'),
            'drive_topic': LaunchConfiguration('drive_topic'),
            'drive_frame_id': LaunchConfiguration('drive_frame_id'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        device_arg,
        deterministic_arg,
        scan_topic_arg,
        drive_topic_arg,
        drive_frame_arg,
        sac_node,
    ])
