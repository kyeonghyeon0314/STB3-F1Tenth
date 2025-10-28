from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tomark',
    maintainer_email='tomark@example.com',
    description='Pure Pursuit controller for F1TENTH autonomous racing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pure_pursuit_controller = f1tenth_control.pure_pursuit_controller:main',
            'sac_cnn_controller = f1tenth_control.sac_cnn_controller:main',
        ],
    },
)
