#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ScanHeaderFix(Node):
    def __init__(self):
        super().__init__('scan_header_fix')
        
        # F1tenth의 /scan 토픽을 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # 헤더가 수정된 스캔을 재발행
        self.publisher = self.create_publisher(
            LaserScan,
            '/scan_fixed',
            10
        )
        
        self.get_logger().info('Scan header fix node started')
        
    def scan_callback(self, msg):
        # 헤더 일관성 수정
        fixed_msg = LaserScan()
        fixed_msg.header = msg.header
        fixed_msg.angle_min = msg.angle_min
        fixed_msg.angle_increment = msg.angle_increment
        fixed_msg.time_increment = msg.time_increment
        fixed_msg.scan_time = msg.scan_time
        fixed_msg.range_min = msg.range_min
        fixed_msg.range_max = msg.range_max
        fixed_msg.ranges = msg.ranges
        fixed_msg.intensities = msg.intensities
        
        # 헤더 일관성 맞추기: angle_max를 실제 데이터 개수에 맞게 조정
        # angle_max = angle_min + angle_increment * (N-1)
        num_readings = len(msg.ranges)
        fixed_msg.angle_max = msg.angle_min + msg.angle_increment * (num_readings - 1)
        
        self.publisher.publish(fixed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanHeaderFix()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()