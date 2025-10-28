#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    """Launch F1TENTH integrated system with SAC-based control and RViz."""

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/sac-cnn-latest.zip',
        description='SAC 정책 체크포인트(.zip) 경로'
    )

    # Gym Bridge Launch (시뮬레이션 환경)
    gym_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('f1tenth_gym_ros'),
                'launch',
                'gym_bridge_launch.py'
            ])
        ),
        launch_arguments={
            'map_path': PathJoinSubstitution([
                FindPackageShare('f1tenth_gym_ros'),
                'maps',
                'track.yaml'
            ])
        }.items()
    )

    # Checkpoint Path Planner Launch (체크포인트 기반 경로 계획) - 2초 지연
    path_planner_launch = TimerAction(
        period=2.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('f1tenth_path_planner'),
                        'launch',
                        'checkpoint_path_planner_launch.py'
                    ])
                )
            )
        ]
    )

    # SAC + CNN 정책 기반 제어 노드
    sac_control_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('f1tenth_control'),
                        'launch',
                        'sac_cnn_launch.py'
                    ])
                ),
                launch_arguments={
                    'model_path': LaunchConfiguration('model_path'),
                    'marker_topic': '/sac_control/marker',
                }.items()
            )
        ]
    )

    # RViz 시각화
    rviz_config = PathJoinSubstitution([
        FindPackageShare('f1tenth_control'),
        'rviz',
        'sac_control.rviz'
    ])
    rviz_launch = TimerAction(
        period=4.5,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='sac_control_rviz',
                arguments=['-d', rviz_config],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        model_path_arg,
        gym_bridge_launch,
        path_planner_launch,
        sac_control_launch,
        rviz_launch,
    ])
