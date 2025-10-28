import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 런치 인수들
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    
    # F1tenth용 설정 파일 경로
    default_params_file = os.path.join(
        get_package_share_directory("f1tenth_slam_toolbox"),
        'config', 
        'f1tenth_mapper_params.yaml'
    )

    # 런치 인수 선언
    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='F1tenth 시뮬레이션에서 실제 시간 사용'
    )
    
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='F1tenth SLAM Toolbox 설정 파일의 전체 경로'
    )

    # map≡odom 항등 TF 발행 (필요시 odom을 요구하는 노드용)
    map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 스캔 헤더 수정 노드 (1080→1081 beam 문제 해결)
    scan_header_fix_node = Node(
        package='f1tenth_slam_toolbox',
        executable='scan_header_fix.py',
        name='scan_header_fix',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # SLAM Toolbox 노드 실행 (TF 발행 비활성화)
    start_async_slam_toolbox_node = Node(
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ],
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        remappings=[
            ('/map', '/slam_map'),  # SLAM 생성 맵을 별도 토픽으로 분리
            ('/scan', '/scan_fixed'),  # 수정된 스캔 데이터 사용
        ]
    )

    # 런치 설명 구성
    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(map_odom_tf)
    ld.add_action(scan_header_fix_node)
    ld.add_action(start_async_slam_toolbox_node)

    return ld