#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Launch F1TENTH integrated system with checkpoint-based path planner"""

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

    # Pure Pursuit Control Launch (차량 제어)
    # 조향각에 따른 적응형 속도 제어로 회전 시 감속 (디폴트 값 사용)
    control_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('f1tenth_control'),
                        'launch',
                        'pure_pursuit_launch.py'
                    ])
                )
            )
        ]
    )

    # SAC + CNN 정책 기반 제어 노드 (Pure Pursuit 이후 6초 지연)
    sac_control_launch = TimerAction(
        period=6.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('f1tenth_control'),
                        'launch',
                        'sac_cnn_launch.py'
                    ])
                )
            )
        ]
    )

    return LaunchDescription([
        gym_bridge_launch,
        path_planner_launch,
        control_launch,
        sac_control_launch,
    ])
