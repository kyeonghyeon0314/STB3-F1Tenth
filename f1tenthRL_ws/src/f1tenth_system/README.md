# f1tenth_system

f1tenth 레이스카 온보드 드라이버. 이 브랜치는 ROS 2로의 마이그레이션을 위해 개발 중입니다. 시작 방법은 [F1TENTH 문서](https://f1tenth.readthedocs.io/en/foxy_test/getting_started/firmware/index.html)를 참고하세요.

## 하위 모듈 클론
이 저장소를 클론할 때 하위 모듈도 함께 클론해야 합니다. 다음 명령을 실행하세요:

```bash
git submodule update --init --recursive --remote
```

이렇게 하면 모든 하위 모듈이 구성된 브랜치로 클론되고 최신 상태로 업데이트됩니다.

## 데드맨 스위치
Logitech F-710 조이스틱에서 LB 버튼은 텔레옵(teleop)용 데드맨 스위치이고, RB 버튼은 내비게이션용 데드맨 스위치입니다. 버튼 매핑은 변경할 수 있습니다. 방법은 readthedocs 문서를 참고하세요.

## Sick Lidar

sick_scan_xd는 ROS 2 Foxy용 apt 패키지가 없습니다. [소스에서 설치](https://github.com/SICKAG/sick_scan_xd)를 시도할 수 있습니다. ROS 2 Humble에서는 다음 단계로 sick_scan_xd를 설치할 수 있습니다:

```
sudo apt update
sudo apt-get install ros-humble-sick-scan-xd
```

**주의:** `f1tenth_stack` 패키지의 launch 디렉터리에 있는 `sick_tim_5xx.launch` 파일을 사용해야 합니다:

`<f1tenth_stack>/launch/sick_tim_5xx.launch`

`ros2 launch f1tenth_stack sick_bringup_launch.py`로 실행합니다. 이 파일은 원래의 `sick_tim_5xx.launch` 파일을 업데이트하여 "frame_id"를 "laser"로, "tf_base_frame_id"를 "base_link"로 변경했기 때문에 slam_toolbox 및 파티클 필터와 호환됩니다. 또한 launch 파일에서 라이다의 IP 주소를 설정해야 합니다.

`ros2 launch f1tenth_stack sick_bringup_launch.py`로 실행합니다.

시작 방법은 [F1TENTH 문서](https://f1tenth.readthedocs.io/en/foxy_test/getting_started/firmware/index.html)를 참고하세요.

### SICK Lidar IP 확인

라이다는 기본 IP 주소로 출고됩니다. IP 주소를 읽거나 변경하는 방법은 두 가지입니다:

1. **Windows 컴퓨터 사용:** [SICK SOPAS ET](https://www.sick.com/de/de/sopas-engineering-tool-2018/p/p367244)를 설치하고, [sick_scan_xd 패키지의 안내](https://github.com/SICKAG/sick_scan_xd?tab=readme-ov-file#starting-with-a-new-lidar)를 따르세요.

2. **Linux 컴퓨터 사용:** 다음 명령으로 네트워크에서 라이다의 IP 주소를 스캔할 수 있습니다:

```bash
nmap -sn
```

네트워크에 SICK 라이다가 연결되어 있다면 출력에서 라이다의 IP 주소가 표시됩니다.

## 토픽

### 드라이버 스택이 구독하는 토픽
- `/drive`: 자율 주행용 토픽, `AckermannDriveStamped` 메시지를 사용합니다.

### 드라이버 스택이 퍼블리시하는 센서 토픽
- `/scan`: `LaserScan` 메시지 토픽.
- `/odom`: `Odometry` 메시지 토픽.
- `/sensors/imu/raw`: `Imu` 메시지 토픽.
- `/sensors/core`: VESC 텔레메트리 데이터 토픽.

## 외부 의존성

1. ackermann_msgs [https://index.ros.org/r/ackermann_msgs/#humble](https://index.ros.org/r/ackermann_msgs/#humble).
2. urg_node [https://index.ros.org/p/urg_node/#humble](https://index.ros.org/p/urg_node/#humble). Hokuyo LiDAR 드라이버입니다.
3. joy [https://index.ros.org/p/joy/#humble](https://index.ros.org/p/joy/#humble). ROS 2의 조이스틱 드라이버입니다.
4. teleop_tools  [https://index.ros.org/p/teleop_tools/#humble](https://index.ros.org/p/teleop_tools/#humble). ROS 2에서 조이스틱 텔레옵 패키지입니다.
5. vesc [GitHub - f1tenth/vesc at ros2](https://github.com/f1tenth/vesc/tree/ros2). ROS 2용 VESC 드라이버입니다.
6. ackermann_mux [GitHub - f1tenth/ackermann_mux: Twist multiplexer](https://github.com/f1tenth/ackermann_mux). ROS 2에서 Ackermann 메시지를 멀티플렉싱하는 패키지입니다.
<!-- 7. rosbridge_suite [https://index.ros.org/p/rosbridge_suite/#humble-overview](https://index.ros.org/p/rosbridge_suite/#humble-overview) ROS 2에서 websocket 연결을 제공하는 패키지입니다. -->

## 이 저장소의 패키지

1. f1tenth_stack: bringup launch 및 모든 파라미터 파일을 관리합니다

## Bringup에서 실행되는 노드

1. joy
2. joy_teleop
3. ackermann_to_vesc_node
4. vesc_to_odom_node
5. vesc_driver_node
6. urg_node
7. ackermann_mux

## 종속 패키지의 파라미터와 토픽

### vesc_driver

1. 파라미터:
   - duty_cycle_min, duty_cycle_max
   - current_min, current_max
   - brake_min, brake_max
   - speed_min, speed_max
   - position_min, position_max
   - servo_min, servo_max
2. 퍼블리시:
   - sensors/core
   - sensors/servo_position_command
   - sensors/imu
   - sensors/imu/raw
3. 구독:
   - commands/motor/duty_cycle
   - commands/motor/current
   - commands/motor/brake
   - commands/motor/speed
   - commands/motor/position
   - commands/servo/position

### ackermann_to_vesc

1. 파라미터:
   - speed_to_erpm_gain
   - speed_to_erpm_offset
   - steering_angle_to_servo_gain
   - steering_angle_to_servo_offset
2. 퍼블리시:
   - ackermann_cmd
3. 구독:
   - commands/motor/speed
   - commands/servo/position

### vesc_to_odom

1. 파라미터:
   - odom_frame
   - base_frame
   - use_servo_cmd_to_calc_angular_velocity
   - speed_to_erpm_gain
   - speed_to_erpm_offset
   - steering_angle_to_servo_gain
   - steering_angle_to_servo_offset
   - wheelbase
   - publish_tf
2. 퍼블리시:
   - odom
3. 구독:
   - sensors/core
   - sensors/servo_position_command

### throttle_interpolator

1. 파라미터:
   - rpm_input_topic
   - rpm_output_topic
   - servo_input_topic
   - servo_output_topic
   - max_acceleration
   - speed_max
   - speed_min
   - throttle_smoother_rate
   - speed_to_erpm_gain
   - max_servo_speed
   - steering_angle_to_servo_gain
   - servo_smoother_rate
   - servo_max
   - servo_min
   - steering_angle_to_servo_offset
2. 퍼블리시:
   - rpm_output_topic에 지정된 토픽
   - servo_output_topic에 지정된 토픽
3. 구독:
   - rpm_input_topic에 지정된 토픽
   - servo_input_topic에 지정된 토픽
