# F1Tenth Gym ROS2 패키지 설정 파일
# ROS2에서 F1Tenth 시뮬레이션을 사용하기 위한 패키지 빌드 및 설치 설정

from setuptools import setup 
import os 
from glob import glob

# 패키지 이름
package_name = 'f1tenth_gym_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # 설치할 데이터 파일들
        # 패키지 리소스 인덱스 등록 (ROS2에서 패키지 발견을 위함)
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # 패키지 메타데이터 파일
        ('share/' + package_name, ['package.xml']),
        # 런치 파일들 (.py 파일)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # URDF/Xacro 파일들 (로봇 모델 정의)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xacro')),
        # RViz 설정 파일들 (시각화 설정)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
        # 설정 파일들 (매개변수 등)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # 맵 파일들 (.yaml, .png 등)
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    # 설치 시 필요한 의존성
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='Bridge for using f1tenth_gym in ROS2',
    license='MIT',
    # 실행 가능한 스크립트 정의
    entry_points={
        'console_scripts': [
            # gym_bridge 명령어 정의
            'gym_bridge = f1tenth_gym_ros.gym_bridge:main'
        ],
    },
)
