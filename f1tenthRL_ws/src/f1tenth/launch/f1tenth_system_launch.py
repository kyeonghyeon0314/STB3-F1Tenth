#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Launch F1TENTH integrated system"""
    
    # Gym Bridge Launch (시뮬레이션 환경)
    gym_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('f1tenth_gym_ros'),
                'launch',
                'gym_bridge_launch.py'
            ])
        )
    )
    
    # Path Planner Launch (경로 계획) - 2초 지연
    path_planner_launch = TimerAction(
        period=2.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('f1tenth_path_planner'),
                        'launch',
                        'path_planner_launch.py'
                    ])
                )
            )
        ]
    )
    
    # Pure Pursuit Control Launch (차량 제어) - 4초 지연
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
    
    return LaunchDescription([
        gym_bridge_launch,
        path_planner_launch,
        control_launch,
    ])