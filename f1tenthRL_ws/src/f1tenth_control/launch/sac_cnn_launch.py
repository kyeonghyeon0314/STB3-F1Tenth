#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/sac-cnn-latest.zip',
        description='Stable Baselines 3 SAC 모델 (.zip) 경로'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
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
        default_value='ego_racecar/base_link',
        description='드라이브 메시지 frame_id'
    )

    marker_topic_arg = DeclareLaunchArgument(
        'marker_topic',
        default_value='/sac_control/marker',
        description='RViz Marker 토픽'
    )

    filtered_scan_topic_arg = DeclareLaunchArgument(
        'filtered_scan_topic',
        default_value='/filtered_scan',
        description='CNN 특징 벡터 퍼블리시 토픽 (64-dim)'
    )

    # Action 역정규화 범위 (학습 시 설정과 동일하게!)
    speed_min_arg = DeclareLaunchArgument(
        'speed_min',
        default_value='0.0',
        description='최소 속도 (m/s) - 학습 시 training_speed_min과 동일'
    )

    speed_max_arg = DeclareLaunchArgument(
        'speed_max',
        default_value='5.0',
        description='최대 속도 (m/s) - 학습 시 --speed-cap과 동일'
    )

    steering_min_arg = DeclareLaunchArgument(
        'steering_min',
        default_value='-1.066',
        description='최소 조향각 (rad)'
    )

    steering_max_arg = DeclareLaunchArgument(
        'steering_max',
        default_value='1.066',
        description='최대 조향각 (rad)'
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
            'marker_topic': LaunchConfiguration('marker_topic'),
            'filtered_scan_topic': LaunchConfiguration('filtered_scan_topic'),
            'speed_min': LaunchConfiguration('speed_min'),
            'speed_max': LaunchConfiguration('speed_max'),
            'steering_min': LaunchConfiguration('steering_min'),
            'steering_max': LaunchConfiguration('steering_max'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        device_arg,
        deterministic_arg,
        scan_topic_arg,
        drive_topic_arg,
        drive_frame_arg,
        marker_topic_arg,
        filtered_scan_topic_arg,
        speed_min_arg,
        speed_max_arg,
        steering_min_arg,
        steering_max_arg,
        sac_node,
    ])
