#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    # Attempt to read f1tenth_gym_ros sim.yaml to reuse map_path/map_img_ext
    params = {}
    try:
        fgr = get_package_share_directory('f1tenth_gym_ros')
        sim_yaml = os.path.join(fgr, 'config', 'sim.yaml')
        with open(sim_yaml, 'r') as f:
            cfg = yaml.safe_load(f)
        bridge = cfg['bridge']['ros__parameters']
        params['map_path'] = bridge['map_path']
        params['map_img_ext'] = bridge['map_img_ext']
        params['map_yaml_path'] = bridge['map_path'] + '.yaml'
    except Exception:
        pass

    global_node = Node(
        package='f1tenth_path_planner',
        executable='global_centerline_node',
        name='global_centerline_node',
        output='screen',
        parameters=[params]
    )

    local_node = Node(
        package='f1tenth_path_planner',
        executable='local_avoidance_node',
        name='local_avoidance_node',
        output='screen',
        parameters=[{
            'global_path_topic': '/global_path',
            'scan_topic': '/scan',  # or '/scan_fixed' if using SLAM fix node
            'local_horizon': 8.0,
            'path_resolution': 0.2,
            'lateral_offsets': [0.0, 0.4, -0.4],
            'safety_radius': 0.4,
            'lookahead_distance': 2.0,
            'base_frame': 'ego_racecar/base_link',
            'map_frame': 'map',
        }]
    )

    ld = LaunchDescription()
    ld.add_action(global_node)
    ld.add_action(local_node)
    return ld

