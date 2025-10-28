# F1tenth 레이싱 시뮬레이션을 위한 키보드 원격 조작 노드
# 키보드 입력을 받아서 cmd_vel 토픽으로 Twist 메시지를 발행
import sys

import geometry_msgs.msg
import rclpy

# 플랫폼별 키보드 입력 처리를 위한 라이브러리
# msvcrt: Windows용 키보드 입력 처리 (Microsoft Visual C Runtime)
# termios: Unix/Linux 터미널 설정 제어 (터미널 속성 저장/복원)
# tty: Unix/Linux raw 모드 키보드 입력 (버퍼링 없이 즉시 입력)
if sys.platform == 'win32':
    import msvcrt
else:
    import termios 
    import tty


# 키보드 조작 안내 메시지
msg = """
F1tenth 키보드 원격조작 노드
키보드 입력을 받아서 Twist 메시지로 변환하여 발행합니다.
미국식 키보드 레이아웃에 최적화되어 있습니다.
---------------------------
이동 제어:
   u    i    o
   j    k    l
   m    ,    .

홀로노믹 모드 (옆으로 이동) - Shift 키와 함께 사용:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : 위로 이동 (+z)
b : 아래로 이동 (-z)

기타 키 : 정지

q/z : 전체 속도 10% 증가/감소
w/x : 선속도만 10% 증가/감소  
e/c : 각속도만 10% 증가/감소

CTRL-C : 프로그램 종료
"""

# 움직임 키 바인딩 - 튜플 값: (linear_x, linear_y, linear_z, angular_z)
moveBindings = {
    'i': (1, 0, 0, 0),    # 전진
    'o': (1, 0, 0, -1),   # 전진 + 우회전
    'j': (0, 0, 0, 1),    # 좌회전
    'l': (0, 0, 0, -1),   # 우회전
    'u': (1, 0, 0, 1),    # 전진 + 좌회전
    ',': (-1, 0, 0, 0),   # 후진
    '.': (-1, 0, 0, 1),   # 후진 + 좌회전
    'm': (-1, 0, 0, -1),  # 후진 + 우회전
    'O': (1, -1, 0, 0),   # 전진 + 우측 이동 (홀로노믹)
    'I': (1, 0, 0, 0),    # 전진 (홀로노믹)
    'J': (0, 1, 0, 0),    # 좌측 이동 (홀로노믹)
    'L': (0, -1, 0, 0),   # 우측 이동 (홀로노믹)
    'U': (1, 1, 0, 0),    # 전진 + 좌측 이동 (홀로노믹)
    '<': (-1, 0, 0, 0),   # 후진 (홀로노믹)
    '>': (-1, -1, 0, 0),  # 후진 + 우측 이동 (홀로노믹)
    'M': (-1, 1, 0, 0),   # 후진 + 좌측 이동 (홀로노믹)
    't': (0, 0, 1, 0),    # 위로 이동 (+z)
    'b': (0, 0, -1, 0),   # 아래로 이동 (-z)
}

# 속도 조절 키 바인딩 - 튜플 값: (선속도 배수, 각속도 배수)
speedBindings = {
    'q': (1.1, 1.1),  # 전체 속도 10% 증가
    'z': (.9, .9),    # 전체 속도 10% 감소
    'w': (1.1, 1),    # 선속도만 10% 증가
    'x': (.9, 1),     # 선속도만 10% 감소
    'e': (1, 1.1),    # 각속도만 10% 증가
    'c': (1, .9),     # 각속도만 10% 감소
}


def getKey(settings):
    """키보드 입력을 받는 함수 (플랫폼별로 다른 방식 사용)"""
    if sys.platform == 'win32':
        # Windows에서는 msvcrt 사용
        key = msvcrt.getwch()
    else:
        # Linux/Unix에서는 termios 사용하여 raw 모드로 키 입력 받기
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    """터미널 설정 저장 (Linux/Unix 전용)"""
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    """터미널 설정 복원 (Linux/Unix 전용)"""
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(speed, turn):
    """현재 속도 설정을 문자열로 반환"""
    return 'currently:\tspeed %s\tturn %s ' % (speed, turn)


def main():
    """메인 함수 - 키보드 입력을 받아 로봇 제어 명령 발행"""
    # 터미널 설정 저장 (종료 시 복원용)
    settings = saveTerminalSettings()

    # ROS2 초기화
    rclpy.init()

    # 노드 생성 및 퍼블리셔 설정
    node = rclpy.create_node('teleop_twist_keyboard')
    pub = node.create_publisher(geometry_msgs.msg.Twist, 'cmd_vel', 10)

    # 초기 속도 및 움직임 변수 설정
    speed = 0.5    # 선속도 스케일링 팩터
    turn = 1.0     # 각속도 스케일링 팩터
    x = 0.0        # 선속도 x 방향
    y = 0.0        # 선속도 y 방향  
    z = 0.0        # 선속도 z 방향
    th = 0.0       # 각속도 z축 (yaw)
    status = 0.0   # 상태 출력 주기 카운터

    try:
        # 사용법 출력
        print(msg)
        print(vels(speed, turn))
        
        # 메인 루프 - 키 입력 처리
        while True:
            key = getKey(settings)
            
            # 움직임 키가 눌린 경우
            if key in moveBindings.keys():
                x = moveBindings[key][0]   # linear.x
                y = moveBindings[key][1]   # linear.y
                z = moveBindings[key][2]   # linear.z
                th = moveBindings[key][3]  # angular.z
                
            # 속도 조절 키가 눌린 경우
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]  # 선속도 조절
                turn = turn * speedBindings[key][1]    # 각속도 조절

                print(vels(speed, turn))
                # 주기적으로 사용법 재출력 (15회마다)
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
                
            # 그 외 키가 눌린 경우 정지
            else:
                x = 0.0
                y = 0.0
                z = 0.0
                th = 0.0
                # Ctrl+C 감지하여 프로그램 종료
                if (key == '\x03'):
                    break

            # Twist 메시지 생성 및 발행
            twist = geometry_msgs.msg.Twist()
            twist.linear.x = x * speed
            twist.linear.y = y * speed
            twist.linear.z = z * speed
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = th * turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        # 프로그램 종료 시 로봇 정지
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

        # 터미널 설정 복원
        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()
