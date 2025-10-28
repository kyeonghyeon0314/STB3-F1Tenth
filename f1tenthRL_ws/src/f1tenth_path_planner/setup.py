# Path planner package setup (ament_python)
from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_path_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21,<1.24',
        'scikit-image>=0.18,<0.22',
    ],
    zip_safe=True,
    maintainer='tomark',
    maintainer_email='tomark@example.com',
    description='Global centerline and LiDAR local avoidance path planner for F1TENTH Gym ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'global_centerline_node = f1tenth_path_planner.global_centerline_node:main',
            'global_checkpoint_node = f1tenth_path_planner.global_checkpoint_node:main',
            'local_avoidance_node = f1tenth_path_planner.local_avoidance_node:main',
            'extract_centerline = f1tenth_path_planner.utils:extract_centerline_cli',
            'checkpoint_recorder_node = f1tenth_path_planner.checkpoint_recorder_node:main',
        ],
    },
)
