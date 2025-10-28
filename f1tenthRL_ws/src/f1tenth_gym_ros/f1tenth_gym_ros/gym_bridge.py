import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

from pathlib import Path
import gymnasium as gym
import numpy as np
from transforms3d import euler

class GymBridge(Node):
    """
    F1TENTH Gym 시뮬레이터와 ROS2를 연결하는 브리지 노드
    - 시뮬레이터에서 센서 데이터를 받아 ROS 토픽으로 퍼블리시
    - ROS 토픽에서 제어 명령을 받아 시뮬레이터에 전달
    - 단일 또는 다중 에이전트 시뮬레이션 지원
    """
    
    def __init__(self):
        super().__init__('gym_bridge')
        
        package_root = Path(__file__).resolve().parents[1]
        default_map_prefix = str(package_root / 'maps' / 'Spielberg_map')

        default_params = {
            'ego_namespace': 'ego_racecar',
            'ego_odom_topic': 'odom',
            'ego_opp_odom_topic': 'opp_odom',
            'ego_scan_topic': 'scan',
            'ego_drive_topic': 'drive',
            'opp_namespace': 'opp_racecar',
            'opp_odom_topic': 'odom',
            'opp_ego_odom_topic': 'opp_odom',
            'opp_scan_topic': 'opp_scan',
            'opp_drive_topic': 'opp_drive',
            'scan_distance_to_base_link': 0.0,
            'scan_fov': 4.7,
            'scan_beams': 1080,
            'map_path': default_map_prefix,
            'map_img_ext': '.png',
            'num_agent': 1,
            'sx': 0.0,
            'sy': 0.0,
            'stheta': 0.0,
            'sx1': 2.0,
            'sy1': 0.5,
            'stheta1': 0.0,
            'kb_teleop': True,
        }

        for name, default in default_params.items():
            self.declare_parameter(name, default)

        self.num_agents = self.get_parameter('num_agent').value

        # 유효성 검사
        if self.num_agents < 1 or self.num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(self.num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # 라이다 센서 설정
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        
        # 메인 차량(ego) 설정
        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_namespace = self.get_parameter('ego_namespace').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        self.ego_drive_published = False
        
        # 상대방 차량(opponent) 설정
        if self.num_agents == 2:
            self.has_opp = True
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_namespace = self.get_parameter('opp_namespace').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.opp_drive_published = False
        else:
            self.has_opp = False
        
        # TF 브로드캐스터 초기화 및 타이머 설정
        self.env = gym.make(
            'f110_gym:f110-v0',
            map=self.get_parameter('map_path').value,
            map_ext=self.get_parameter('map_img_ext').value,
            num_agents=self.num_agents)

        if self.num_agents == 2:
            initial_poses = np.array([[sx, sy, stheta], [sx1, sy1, stheta1]], dtype=float)
        else:
            initial_poses = np.array([[sx, sy, stheta]], dtype=float)

        self._reset_environment(initial_poses)

        # TF 브로드캐스터
        self.br = TransformBroadcaster(self)
        # 타이머: 시뮬레이션 스텝(100Hz) / 퍼블리시 루프(250Hz)
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)
        self.timer = self.create_timer(0.004, self.timer_callback)
        
        # 토픽 이름 설정
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        ego_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_odom_topic').value

        if self.has_opp:
            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_odom_topic').value
            ego_opp_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_ego_odom_topic').value

        # 퍼블리셔
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        
        if self.has_opp:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 10)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 10)

        # 서브스크라이버
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped, ego_drive_topic, self.drive_callback, 10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.ego_reset_callback, 10)
        
        if self.has_opp:
            opp_drive_topic = self.get_parameter('opp_drive_topic').value
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped, opp_drive_topic, self.opp_drive_callback, 10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped, '/goal_pose', self.opp_reset_callback, 10)
        
        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist, '/cmd_vel', self.teleop_callback, 10)
    
    def drive_callback(self, drive_msg):
        """
        ego 차량 제어 명령 콜백
        - 목적: ego 차량의 목표 속도/조향각을 갱신
        - 입력: AckermannDriveStamped (speed [m/s], steering_angle [rad])
        """
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        """
        opp 차량 제어 명령 콜백 (다중 에이전트일 때만)
        - 목적: opp 차량의 목표 속도/조향각을 갱신
        - 입력: AckermannDriveStamped (speed [m/s], steering_angle [rad])
        """
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def teleop_callback(self, twist_msg):
        """
        키보드 텔레오퍼레이션 콜백 (옵션)
        - 목적: /cmd_vel을 통해 간단한 전/후진 및 좌/우 조향을 ego에 반영
        - 입력: geometry_msgs/Twist (linear.x [m/s], angular.z [rad])
        - 텔레오프 사용 시에도 정식 Ackermann 명령(조향 모델 제어)과 동일하게 drive_timer에서 스텝이 진행됨
        """
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def _reset_environment(self, poses):
        """
        Gymnasium API helper to reset simulator state while keeping cached scans in sync.
        """
        self.obs, _ = self.env.reset(options={'poses': np.asarray(poses, dtype=float)})
        self.terminated = False
        self.truncated = False
        self._update_sim_state()

    def ego_reset_callback(self, pose_msg):
        """
        ego 차량 위치 리셋 콜백
        - 목적: RViz 등의 /initialpose 입력을 받아 ego(및 필요 시 opp)의 초기 포즈로 시뮬레이터 리셋
        - 입력: geometry_msgs/PoseWithCovarianceStamped (x, y, quaternion)
        - Rviz에서 초기 Pose 설정 시, ego 차량을 이동
        - 좌표/프레임: 맵 좌표계(map) 기준 포즈로 가정
        """
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            poses = np.array([[rx, ry, rtheta], opp_pose], dtype=float)
        else:
            poses = np.array([[rx, ry, rtheta]], dtype=float)
        self._reset_environment(poses)

    def opp_reset_callback(self, pose_msg):
        """
        opp 차량 위치 리셋 콜백 (다중 에이전트일 때만)
        - 목적: /goal_pose 입력을 받아 opp 포즈만 갱신하여 env.reset 수행
        - 입력: geometry_msgs/PoseStamped
        - Rivz에서 Goal 설정 시, opp 차량을 이동
        """
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            poses = np.array([list(self.ego_pose), [rx, ry, rtheta]], dtype=float)
            self._reset_environment(poses)

    
    def drive_timer_callback(self):
        """
        물리 시뮬레이션 스텝 실행 (주기 타이머)
        - 목적: 현재 명령(ego/opp)을 바탕으로 env.step을 1스텝 진행
        - 주기: 0.01s (100 Hz)
        - (ego_drive_published 플래그) 최초 명령 수신 전에는 스텝을 진행 X
        """
        if self.ego_drive_published and not self.has_opp:
            step_action = np.array([[self.ego_steer, self.ego_requested_speed]], dtype=np.float32)
            self.obs, _, self.terminated, self.truncated, _ = self.env.step(step_action)
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            step_action = np.array([[self.ego_steer, self.ego_requested_speed],
                                    [self.opp_steer, self.opp_requested_speed]], dtype=np.float32)
            self.obs, _, self.terminated, self.truncated, _ = self.env.step(step_action)
        self._update_sim_state()

    def timer_callback(self):
        """
        LiDAR/Odom/TF 퍼블리싱 루프 (주기 타이머)
        - 목적: 현재 시뮬레이션 상태를 ROS 토픽/TF로 퍼블리시합니다.
        - 주기: 0.004s (≈250 Hz)
        - 퍼블리시:
          * LaserScan: ego(+opp)
          * Odometry: ego(+opp), 상호 추정 odom(ego_opp/opp_ego)
          * TF: map→base_link, base_link→laser, 휠 힌지→휠(조향각)
        - 타임스탬프: 노드 clock 기준 now()
        """
        ts = self.get_clock().now().to_msg()

        self._publish_laser_scan(ts)
        self._publish_odom(ts)
        self._publish_transforms(ts)
        self._publish_laser_transforms(ts)
        self._publish_wheel_transforms(ts)

    
    def _update_sim_state(self):
        """
        시뮬레이터 관측값을 내부 상태로 반영
        - 목적: env.step/reset 결과 -> self.obs (gym 관측 dict) -> 포즈/속도/스캔 버퍼
        - 이 함수는 순수 상태 동기화만 수행
        """
        self.ego_scan = self.obs['scans'][0].astype(float).tolist()
        self.ego_pose[0] = float(self.obs['poses_x'][0])
        self.ego_pose[1] = float(self.obs['poses_y'][0])
        self.ego_pose[2] = float(self.obs['poses_theta'][0])
        self.ego_speed[0] = float(self.obs['linear_vels_x'][0])
        self.ego_speed[1] = float(self.obs['linear_vels_y'][0])
        self.ego_speed[2] = float(self.obs['ang_vels_z'][0])
        
        if self.has_opp:
            self.opp_scan = self.obs['scans'][1].astype(float).tolist()
            self.opp_pose[0] = float(self.obs['poses_x'][1])
            self.opp_pose[1] = float(self.obs['poses_y'][1])
            self.opp_pose[2] = float(self.obs['poses_theta'][1])
            self.opp_speed[0] = float(self.obs['linear_vels_x'][1])
            self.opp_speed[1] = float(self.obs['linear_vels_y'][1])
            self.opp_speed[2] = float(self.obs['ang_vels_z'][1])
    
    def _publish_laser_scan(self, ts):
        """
        라이다 스캔 데이터 퍼블리시
        - 목적: 최신 스캔 버퍼를 LaserScan 메시지로 송신
        - 프레임: {namespace}/laser
        """
        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

    def _publish_odom(self, ts):
        """
        오도메트리 데이터 퍼블리시
        - 목적: 현재 포즈/속도를 nav_msgs/Odometry로 송신
        - 교차 퍼블리시: ego_opp_odom_pub / opp_ego_odom_pub (상대 차량에 대한 odom 제공)
        - covariance는 설정하지 않으므로 다운스트림에서 필요 시 주의 요망
        """
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        """
        차량 베이스 링크 TF 퍼블리시
        - 목적: map → {namespace}/base_link 변환을 TF로 브로드캐스트
        - 레이저/휠 등 하위 링크는 별도 함수에서 송신
        """
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = euler.euler2quat(0.0, 0.0, self.opp_pose[2], axes='sxyz')
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        """
        바퀴 조향각 TF 퍼블리시
        - 목적: 조향각을 반영하여 힌지→휠 링크 회전 TF를 브로드캐스트
        - 순수 회전만 갱신(translation 없음)
        """
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)

    def _publish_laser_transforms(self, ts):
        """
        라이다 센서 TF 퍼블리시
        - 목적: base_link → laser 고정 변환을 송신
        """
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        self.br.sendTransform(ego_scan_ts)

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = self.scan_distance_to_base_link
            opp_scan_ts.transform.rotation.w = 1.
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            self.br.sendTransform(opp_scan_ts)


def main(args=None):
    """
    메인 함수 - 노드 초기화 및 실행
    - 목적: rclpy 초기화, GymBridge 노드 생성, 스핀 진입
    - 종료: Ctrl+C 등으로 스핀 종료 시 rclpy.shutdown()은 런타임이 정리
    - 외부에서 launch 파일로 여러 노드를 구동할 경우, 파라미터를 launch에서 전달
    """
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()
