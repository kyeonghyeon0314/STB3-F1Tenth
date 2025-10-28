# F1Tenth Gym 브리지 런치 파일
# F1Tenth 시뮬레이션 환경을 ROS2에서 실행하기 위한 모든 노드들을 시작

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # 런치 설명 객체 생성
    ld = LaunchDescription()

    # 맵 YAML 파일 경로 인자 선언
    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value='',
        description='Path to the map YAML file (with .yaml extension). If empty, uses the map from sim.yaml config file.'
    )
    
    # 맵 경로 런치 설정 가져오기
    map_path_config = LaunchConfiguration('map_path')

    # 설정 파일
    ## 경로 구성
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml'
        )
    ## 파싱
    config_dict = yaml.safe_load(open(config, 'r'))

    # 상대 차량 존재 여부
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1
    # 키보드 조작 여부
    teleop = config_dict['bridge']['ros__parameters']['kb_teleop']

    # 맵 경로 결정 (런치 인자가 제공되면 사용, 아니면 설정 파일 사용)
    # LaunchConfiguration은 런타임에 평가되므로 직접 비교 불가
    # 따라서 PythonExpression과 OpaqueFunction 사용 필요
    from launch.actions import OpaqueFunction

    def setup_nodes_with_map(context):
        # 런치 인자에서 맵 경로 가져오기
        map_path_arg_value = context.launch_configurations.get('map_path', '')

        # 인자가 비어있으면 설정 파일에서 가져오기
        if not map_path_arg_value:
            map_path_yaml = config_dict['bridge']['ros__parameters']['map_path'] + '.yaml'
            map_path_base = config_dict['bridge']['ros__parameters']['map_path']
        else:
            # 인자로 받은 경로는 .yaml 확장자 포함
            map_path_yaml = map_path_arg_value
            # .yaml 확장자 제거하여 베이스 경로 추출
            map_path_base = map_path_arg_value.replace('.yaml', '')

        # gym_bridge 노드 - 맵 경로 파라미터 오버라이드
        bridge_node = Node(
            package='f1tenth_gym_ros',
            executable='gym_bridge',
            name='bridge',
            parameters=[config, {'map_path': map_path_base}]
        )

        # map_server 노드
        map_server_node = Node(
            package='nav2_map_server',
            executable='map_server',
            parameters=[{'yaml_filename': map_path_yaml},
                        {'topic': 'map'},
                        {'frame_id': 'map'},
                        {'output': 'screen'},
                        {'use_sim_time': False}]
        )

        return [bridge_node, map_server_node]
    
    # RViz 시각화 노드
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')]
    )

    # Nav2 라이프사이클 관리자 (지도 서버 생명주기 관리)
    ## Nav2: ROS2용 자율주행 네비게이션 프레임워크 (경로계획, 장애물회피, 위치추정 등을 제공)
    ### 현재 패키지에서는 정적 지도 제공만 (map_server)
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': False},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )
    
    # 자아 차량 로봇 상태 퍼블리셔 (URDF 모델 게시)
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'ego_racecar.xacro')])}],
        remappings=[('/robot_description', 'ego_robot_description')]
    )
    
    # 상대 차량 로봇 상태 퍼블리셔 (멀티 에이전트용)
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'opp_racecar.xacro')])}],
        remappings=[('/robot_description', 'opp_robot_description')]
    )

    # 노드들을 런치 설명에 추가
    ## 맵 경로 인자 추가
    ld.add_action(map_path_arg)
    ## 시각화
    ld.add_action(rviz_node)
    ## 시뮬레이션 브리지 및 지도 서버 (OpaqueFunction을 통해 동적으로 생성)
    ld.add_action(OpaqueFunction(function=setup_nodes_with_map))
    ## 라이프사이클 관리
    ld.add_action(nav_lifecycle_node)
    ## 자아 차량
    ld.add_action(ego_robot_publisher)

    ## 상대 차량이 있는 경우에만 추가
    if has_opp:
        ld.add_action(opp_robot_publisher)

    return ld