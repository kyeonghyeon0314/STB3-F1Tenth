# F1TENTH Control Package

F1TENTH 자율주행 레이싱을 위한 Pure Pursuit 기반 차량 제어 패키지입니다.

## 📋 개요

이 패키지는 Pure Pursuit 알고리즘을 사용하여 F1TENTH 차량의 조향을 제어합니다. Local path를 우선적으로 사용하며, 사용할 수 없는 경우 Global path를 백업으로 사용합니다.

<div align="center">
  <img src="../img/control.gif" alt="Pure Pursuit Control Demo" width="500">
  <p><em>Pure Pursuit 제어 시스템 동작 데모</em></p>
</div>

## 🏗️ 아키텍처

```
f1tenth_planning → /local_path  (우선순위)
                → /global_path  (백업)
                    ↓
f1tenth_control → Pure Pursuit 조향 제어
                → 고정 속도 제어
                    ↓
                 /drive (AckermannDriveStamped)
```

## 🔧 주요 기능

### 1. **듀얼 Path 지원**
- **Primary**: `/local_path` 사용 (실시간 장애물 회피)
- **Fallback**: `/global_path` 사용 (안전성 보장)
- **Timeout**: Local path 1초 이상 미수신 시 Global path로 전환

### 2. **Pure Pursuit 제어**
- **Lookahead Distance**: 2.5m (매개변수 조정 가능)
- **조향 제한**: ±24도 (0.4189 rad)
- **제어 주기**: 50Hz
- **차량 파라미터**: F1TENTH 휠베이스 (0.3302m)

### 3. **안전 기능**
- Path/위치 정보 없을 시 자동 정지
- 최대 조향각 제한
- Target point 탐색 실패 시 안전 정지

## 🎯 토픽 인터페이스

### 구독 토픽
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/local_path` | `nav_msgs/Path` | 지역 경로 (우선 사용) |
| `/global_path` | `nav_msgs/Path` | 전역 경로 (백업 사용) |
| `/ego_racecar/odom` | `nav_msgs/Odometry` | 차량 위치/자세 정보 |

### 발행 토픽
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | 차량 제어 명령 |

## 🚀 실행 방법

### 기본 실행
```bash
ros2 launch f1tenth_control pure_pursuit_launch.py
```

### 매개변수 커스터마이징
```bash
ros2 launch f1tenth_control pure_pursuit_launch.py \
    lookahead_distance:=3.0 \
    speed:=3.5 \
    max_steering_angle:=0.5
```

### 토픽 재매핑
```bash
ros2 launch f1tenth_control pure_pursuit_launch.py \
    odom_topic:=/custom/odom \
    drive_topic:=/custom/drive
```

## ⚙️ 매개변수

| 매개변수 | 기본값 | 설명 |
|----------|--------|------|
| `lookahead_distance` | 2.5 | Pure Pursuit 전방 추적 거리 (m) |
| `speed` | 2.0 | 고정 차량 속도 (m/s) |
| `wheelbase` | 0.3302 | F1TENTH 휠베이스 길이 (m) |
| `max_steering_angle` | 0.4189 | 최대 조향각 (rad, ~24°) |
| `path_topic` | `/local_path` | 주요 경로 토픽 |
| `fallback_path_topic` | `/global_path` | 백업 경로 토픽 |
| `odom_topic` | `/ego_racecar/odom` | 오도메트리 토픽 |
| `drive_topic` | `/drive` | 제어 명령 토픽 |

## 🧮 Pure Pursuit 알고리즘

### 수학적 공식
```
δ = atan2(2L * sin(α), ld)
```
- **δ**: 조향각
- **L**: 휠베이스 길이
- **α**: 차량 헤딩과 목표점 사이의 각도
- **ld**: 실제 lookahead 거리

### 알고리즘 단계
1. **목표점 탐색**: Lookahead 거리에서 전방 추적점 찾기
2. **좌표 변환**: 글로벌 좌표 → 차량 좌표계
3. **각도 계산**: 차량 헤딩 대비 목표 방향각 계산
4. **조향각 계산**: Pure Pursuit 공식 적용
5. **안전 제한**: 최대 조향각 범위 내 클램핑

## 🔄 Path 선택 로직

```python
def get_current_path():
    # 1순위: Local Path (최신 + 유효)
    if local_path_is_recent_and_valid:
        return local_path
    
    # 2순위: Global Path (백업)
    if global_path_is_valid:
        return global_path
    
    return None  # 정지
```

## 🛡️ 안전 메커니즘

1. **Path 검증**: 유효한 path가 없으면 정지
2. **목표점 검증**: Target point 탐색 실패 시 정지  
3. **조향 제한**: 하드웨어 한계 내 조향각 제한
4. **거리 검증**: 목표점이 너무 가까우면 직진

## 🐛 디버깅

### 로그 확인
```bash
ros2 launch f1tenth_control pure_pursuit_launch.py --ros-args --log-level debug
```

### 토픽 모니터링
```bash
# 제어 명령 확인
ros2 topic echo /drive

# Path 상태 확인  
ros2 topic hz /local_path
ros2 topic hz /global_path

# 차량 위치 확인
ros2 topic echo /ego_racecar/odom
```

### RViz 시각화
```bash
rviz2 -d $(ros2 pkg prefix f1tenth_control)/share/f1tenth_control/rviz/control_viz.rviz
```

## 🔧 튜닝 가이드

### Lookahead Distance
- **큰 값**: 부드러운 주행, 코너 컷팅 위험
- **작은 값**: 정확한 추종, 진동 가능성

### 속도 설정  
- **높은 속도**: 빠른 주행, 안정성 저하
- **낮은 속도**: 안정적 주행, 성능 제한

### 조향 제한
- F1TENTH 하드웨어 한계 준수
- 과도한 조향은 미끄러짐 유발

## 📊 성능 특성

- **제어 주기**: 50Hz (20ms)
- **반응 지연**: < 50ms
- **메모리 사용량**: ~10MB
- **CPU 사용률**: ~5% (단일 코어)

## 🤝 의존성

### ROS2 패키지
- `rclpy`: Python ROS2 클라이언트
- `geometry_msgs`: 기하학 메시지
- `nav_msgs`: 네비게이션 메시지  
- `ackermann_msgs`: Ackermann 구동 메시지
- `tf2_ros`: 좌표 변환

### F1TENTH 패키지
- `f1tenth_planning`: 경로 계획 (Local/Global path 제공)
- `f1tenth_gym_ros`: 시뮬레이션 환경

## 📈 향후 개선사항

- [ ] 적응적 속도 제어 (곡률 기반)
- [ ] 동적 lookahead 거리 조정
- [ ] 예측 제어 (MPC) 통합
- [ ] 다중 경로 융합 알고리즘
- [ ] 실시간 매개변수 튜닝 인터페이스

---

**📝 Note**: 이 패키지는 시뮬레이션 환경에 최적화되어 있으며, 실제 하드웨어 적용 시 매개변수 조정이 필요할 수 있습니다.