# F1TENTH Gym (ROS2 Humble 호환 버전)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

F1TENTH 1/10 스케일 자율 주행 레이싱 플랫폼을 위한 OpenAI Gymnasium 기반 물리 시뮬레이션 환경입니다. 이 버전은 **ROS2 Humble과 호환되도록 Gymnasium API로 마이그레이션**되었습니다.

## 주요 기능

### 물리 시뮬레이션
- **차량 동역학 모델**: Single-Track Dynamic (STD) 모델과 Kinematic (KS) 모델 지원
- **정확한 타이어 모델**: 코너링 강성, 슬립 각도 등 실제 물리 특성 반영
- **적분 방법**: RK4 (기본값) 및 Euler 적분기 선택 가능
- **설정 가능한 물리 파라미터**: 질량, 관성, 마찰 계수, 조향 제한 등

### 다중 에이전트 지원
- 동일한 환경에서 여러 차량 동시 시뮬레이션
- 차량 간 충돌 감지 (GJK 알고리즘)
- 에이전트 간 레이캐스팅으로 LIDAR 스캔 가림 효과 구현
- 에고 에이전트 인덱스 설정 가능

### LIDAR 시뮬레이션
- **ScanSimulator2D**: 효율적인 2D 레이캐스팅 기반 레이저 스캔 시뮬레이션
- 기본 설정: 1080개 빔, 270도 시야각
- 거리 변환 기반 빠른 연산 (scipy.ndimage.edt)
- 장애물 및 다른 차량에 의한 가림 효과
- 최대 범위 및 공간 해상도 설정 가능

### 충돌 감지
- **GJK (Gilbert-Johnson-Keerthi) 알고리즘**: 볼록 다각형 충돌 감지
- 차량 간 충돌 및 환경 충돌 감지
- iTTC (Instantaneous Time-To-Collision) 체크
- 충돌 시 차량 정지

### 시각화
- Pyglet 기반 OpenGL 렌더링
- 두 가지 렌더 모드: `human` (약 200 FPS), `human_fast` (최적화된 고속 렌더링)
- 실시간 랩 타임 및 랩 수 표시
- 사용자 정의 렌더링 콜백 지원
- 카메라 줌 및 팬 기능

## 설치 방법

### 사전 요구사항
- Python 3.8 이상 (ROS2 Humble의 경우 Python 3.10 권장)
- pip 패키지 관리자

### 방법 1

```bash
# 저장소 클론 및 설치
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

### 방법 2: Docker 사용

```bash
# Docker 이미지 빌드
docker build -t f1tenth_gym -f Dockerfile .

# 컨테이너 실행
docker run -it --name=f1tenth_gym_container --rm f1tenth_gym

# GPU 지원 및 디스플레이 사용 시
docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  f1tenth_gym
```

### 방법 3: pip 직접 설치

```bash
pip install git+https://github.com/t0mark/f1tenth_gym_ros.git
```

### 의존성 패키지

- `gymnasium>=0.28.1` - OpenAI Gym의 현대적인 대체품
- `numpy>=1.21.0,<1.25.0` - 수치 연산
- `Pillow>=9.0.1` - 맵 이미지 로딩
- `scipy>=1.7.3` - 거리 변환 (EDM)
- `numba>=0.56.4,<0.59.0` - JIT 컴파일을 통한 물리 시뮬레이션 가속
- `pyyaml>=5.3.1` - 맵 설정 파일 파싱
- `pyglet<1.6` - OpenGL 렌더링
- `pyopengl` - OpenGL 바인딩

## 빠른 시작

### 기본 사용법

```python
import gymnasium as gym
import numpy as np

# 환경 생성
env = gym.make('f110_gym:f110-v0',
               map="berlin",
               num_agents=1,
               timestep=0.01,
               render_mode='human')

# 초기 포즈 설정 (x, y, theta)
initial_pose = np.array([[0.0, 0.0, 0.0]])

# 환경 리셋
obs, info = env.reset(options={'poses': initial_pose})

# 시뮬레이션 루프
done = False
while not done:
    # 액션: [조향각(rad), 속도(m/s)]
    action = np.array([[0.0, 1.0]])

    # 한 스텝 진행
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 렌더링
    env.render()

env.close()
```

### 다중 에이전트 예제

```python
import gymnasium as gym
import numpy as np

# 2개의 에이전트로 환경 생성
env = gym.make('f110_gym:f110-v0',
               map="berlin",
               num_agents=2,
               render_mode='human')

# 각 에이전트의 초기 포즈
poses = np.array([
    [0.0, 0.0, 0.0],      # 에이전트 0 (에고)
    [2.0, 0.0, 0.0]       # 에이전트 1
])

obs, info = env.reset(options={'poses': poses})

done = False
while not done:
    # 각 에이전트의 액션
    actions = np.array([
        [0.1, 1.5],   # 에이전트 0: 조향 0.1 rad, 속도 1.5 m/s
        [0.0, 1.0]    # 에이전트 1: 직진, 속도 1.0 m/s
    ])

    obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    env.render()

env.close()
```

## 사용 예제

### Waypoint Following 예제

저장소의 [examples/waypoint_follow.py](examples/waypoint_follow.py)에서 Pure Pursuit 컨트롤러를 사용한 완전한 예제를 확인할 수 있습니다.

```bash
cd examples
python waypoint_follow.py
```

이 예제는 다음을 포함합니다:
- CSV 파일에서 waypoint 로딩
- Pure Pursuit 경로 추종 알고리즘
- 랩 타임 추적
- 사용자 정의 렌더링 콜백

## 환경 구성

### 차량 파라미터 커스터마이징

```python
import gymnasium as gym

# 커스텀 차량 파라미터
custom_params = {
    'mu': 1.0489,          # 마찰 계수
    'C_Sf': 4.718,         # 전륜 코너링 강성
    'C_Sr': 5.4562,        # 후륜 코너링 강성
    'lf': 0.15875,         # 무게 중심에서 전륜까지 거리 (m)
    'lr': 0.17145,         # 무게 중심에서 후륜까지 거리 (m)
    'h': 0.074,            # 무게 중심 높이 (m)
    'm': 3.74,             # 질량 (kg)
    'I': 0.04712,          # 관성 모멘트 (kg*m^2)
    's_min': -0.4189,      # 최소 조향각 (rad)
    's_max': 0.4189,       # 최대 조향각 (rad)
    'sv_min': -3.2,        # 최소 조향 속도 (rad/s)
    'sv_max': 3.2,         # 최대 조향 속도 (rad/s)
    'v_switch': 7.319,     # 동역학 모델 전환 속도 (m/s)
    'a_max': 9.51,         # 최대 가속도 (m/s^2)
    'v_min': -5.0,         # 최소 속도 (m/s)
    'v_max': 20.0,         # 최대 속도 (m/s)
    'width': 0.31,         # 차량 너비 (m)
    'length': 0.58         # 차량 길이 (m)
}

env = gym.make('f110_gym:f110-v0', params=custom_params)
```

### 환경 설정 옵션

```python
from f110_gym.envs.base_classes import Integrator

env = gym.make('f110_gym:f110-v0',
    # 맵 설정
    map='berlin',              # 맵 이름 (또는 경로)
    map_ext='.png',            # 맵 파일 확장자 (.png 또는 .pgm)

    # 시뮬레이션 설정
    num_agents=1,              # 에이전트 수
    timestep=0.01,             # 물리 시뮬레이션 타임스텝 (초)
    ego_idx=0,                 # 에고 에이전트 인덱스
    integrator=Integrator.RK4, # 적분 방법 (RK4 또는 Euler)

    # 센서 설정
    lidar_dist=0.0,            # LIDAR 높이 오프셋 (미터)

    # 렌더링 설정
    render_mode='human'        # 'human' 또는 'human_fast'
)
```

## API 레퍼런스

### Observation Space

환경은 딕셔너리 형태의 observation을 반환합니다:

```python
obs = {
    'ego_idx': int,                    # 에고 에이전트 인덱스
    'scans': np.ndarray,              # LIDAR 스캔 (num_agents, 1080)
    'poses_x': np.ndarray,            # X 위치 (num_agents,)
    'poses_y': np.ndarray,            # Y 위치 (num_agents,)
    'poses_theta': np.ndarray,        # 요 각도 (num_agents,)
    'linear_vels_x': np.ndarray,      # 종방향 속도 (num_agents,)
    'linear_vels_y': np.ndarray,      # 횡방향 속도 (num_agents,) [항상 0]
    'ang_vels_z': np.ndarray,         # 요 레이트 (num_agents,)
    'collisions': np.ndarray,         # 충돌 플래그 (num_agents,)
    'lap_times': np.ndarray,          # 현재 랩 시간 (num_agents,)
    'lap_counts': np.ndarray,         # 완주한 랩 수 (num_agents,)
}
```

### Action Space

액션은 각 에이전트당 2차원 벡터입니다:
- `action[i][0]`: 조향각 (라디안)
- `action[i][1]`: 속도 (m/s)

```python
# 단일 에이전트
action = np.array([[steering_angle, velocity]])

# 다중 에이전트
action = np.array([
    [steer_0, vel_0],
    [steer_1, vel_1],
    ...
])
```

### 주요 메서드

#### `reset(seed=None, options=None)`

환경을 초기화하고 초기 observation을 반환합니다.

**Parameters:**
- `seed` (int, optional): 재현성을 위한 시드값
- `options` (dict, optional):
  - `poses` (np.ndarray): 초기 포즈 배열 (num_agents, 3) [x, y, theta]

**Returns:**
- `obs` (dict): 초기 observation
- `info` (dict): 추가 정보

```python
obs, info = env.reset(seed=42, options={'poses': initial_poses})
```

#### `step(action)`

액션을 실행하고 한 스텝 진행합니다.

**Parameters:**
- `action` (np.ndarray): 액션 배열 (num_agents, 2)

**Returns:**
- `obs` (dict): 현재 observation
- `reward` (float): 리워드 (현재 타임스텝)
- `terminated` (bool): 에피소드 종료 여부 (충돌 등)
- `truncated` (bool): 시간 제한 초과 여부
- `info` (dict): 추가 정보

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

#### `render()`

환경을 시각화합니다. `render_mode`가 설정된 경우에만 작동합니다.

```python
env.render()
```

#### `add_render_callback(callback_func)`

사용자 정의 렌더링 콜백을 추가합니다.

```python
def custom_render(env_renderer):
    # 커스텀 그래픽 그리기
    pass

env.add_render_callback(custom_render)
```

## ROS2 Humble 호환성

이 버전은 ROS2 Humble과 호환되도록 다음과 같이 수정되었습니다:

### 주요 변경사항

#### 1. Gymnasium 마이그레이션
```python
# 이전: gym 라이브러리
import gym

# 현재: gymnasium 라이브러리
import gymnasium as gym
from gymnasium import spaces
```

#### 2. API 변경사항

**reset() 메서드:**
```python
# 이전 (gym 0.19.0)
obs, reward, done, info = env.reset(poses)

# 현재 (gymnasium 0.28.1+)
obs, info = env.reset(seed=seed, options={'poses': poses})
```

**step() 메서드:**
```python
# 이전
obs, reward, done, info = env.step(action)

# 현재
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

#### 3. Numba JIT 캐싱 비활성화
```python
# 모든 모듈에서
# 이전: @njit(cache=True)
# 현재: @njit(cache=False)
```

이는 최신 Numba 버전과의 호환성 문제를 해결합니다.

#### 4. 타입 일관성
- 모든 observation을 `np.float32`로 통일
- Action space를 명시적으로 float32로 지정
- ROS2 메시지 직렬화 효율성 향상

### ROS2와 통합하기

현재 이 패키지는 네이티브 ROS2 패키지가 아니지만, ROS2 노드와 쉽게 통합할 수 있습니다:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import gymnasium as gym
import numpy as np

class F1TenthSimNode(Node):
    def __init__(self):
        super().__init__('f1tenth_sim_node')

        # 환경 생성
        self.env = gym.make('f110_gym:f110-v0',
                           num_agents=1,
                           render_mode='human')

        # ROS2 Publishers
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # ROS2 Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)

        # 타이머
        self.timer = self.create_timer(0.01, self.sim_step)

        # 상태 초기화
        self.obs, _ = self.env.reset()
        self.action = np.array([[0.0, 0.0]])

    def cmd_callback(self, msg):
        # Twist 메시지를 액션으로 변환
        steering = msg.angular.z  # 조향
        velocity = msg.linear.x   # 속도
        self.action = np.array([[steering, velocity]])

    def sim_step(self):
        # 시뮬레이션 스텝
        self.obs, reward, terminated, truncated, info = \
            self.env.step(self.action)

        # LIDAR 스캔 퍼블리시
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser'
        scan_msg.ranges = self.obs['scans'][0].tolist()
        self.scan_pub.publish(scan_msg)

        # 렌더링
        self.env.render()

        # 종료 시 리셋
        if terminated or truncated:
            self.obs, _ = self.env.reset()

def main(args=None):
    rclpy.init(args=args)
    node = F1TenthSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 맵 및 트랙

### 기본 제공 맵

환경에는 다음 트랙이 포함되어 있습니다:

- **Berlin** - `berlin.png` / `berlin.yaml`
- **Vegas** - `vegas.png` / `vegas.yaml`
- **Skirk** - `skirk.png` / `skirk.yaml`
- **Levine** - `levine.pgm` / `levine.yaml`
- **Stata Basement** - `stata_basement.png` / `stata_basement.yaml`

### 맵 사용법

```python
# 이름으로 사용
env = gym.make('f110_gym:f110-v0', map='berlin', map_ext='.png')

# 또는 경로로 사용
env = gym.make('f110_gym:f110-v0',
               map='/path/to/custom_map',
               map_ext='.png')
```

### 커스텀 맵 생성

맵은 YAML 설정 파일과 이미지 파일로 구성됩니다.

**맵 이미지:**
- PNG 또는 PGM 형식
- 흰색 = 자유 공간, 검은색 = 장애물
- 권장 해상도: 0.05m/픽셀

**YAML 설정 파일 예제:**
```yaml
# custom_map.yaml
image: custom_map.png       # 맵 이미지 파일명
resolution: 0.050000        # 미터/픽셀 비율
origin: [0.0, 0.0, 0.0]    # 맵 원점 [x, y, theta]
negate: 0                   # 이미지 색상 반전 여부 (0 또는 1)
occupied_thresh: 0.65       # 장애물 임계값
free_thresh: 0.196          # 자유 공간 임계값
```

맵 파일은 `gym/f110_gym/envs/maps/` 디렉토리에 위치해야 합니다.

### 랜덤 트랙 생성

`gym/f110_gym/unittest/random_trackgen.py`를 사용하여 임의의 트랙을 생성할 수 있습니다:

```bash
# 추가 의존성 설치
pip install opencv-python shapely matplotlib

# 랜덤 트랙 생성
cd gym/f110_gym/unittest
python random_trackgen.py
```

## 개발 정보

### 프로젝트 구조

```
f1tenth_gym/
├── gym/
│   └── f110_gym/
│       ├── envs/               # 핵심 환경 구현
│       │   ├── f110_env.py     # 메인 Gymnasium 환경
│       │   ├── base_classes.py # RaceCar, Simulator 클래스
│       │   ├── dynamic_models.py # 차량 물리 모델
│       │   ├── laser_models.py  # LIDAR 시뮬레이션
│       │   ├── collision_models.py # 충돌 감지
│       │   ├── rendering.py    # Pyglet 렌더링
│       │   └── maps/           # 맵 파일들
│       └── unittest/           # 단위 테스트
├── examples/                   # 사용 예제
├── docs/                       # 문서화
└── setup.py                    # 패키지 설정
```

### 테스트 실행

```bash
# 동역학 테스트
python gym/f110_gym/unittest/dynamics_test.py

# 충돌 감지 테스트
python gym/f110_gym/unittest/collision_checks.py

# LIDAR 스캔 테스트
python gym/f110_gym/unittest/scan_sim.py

# 렌더링 테스트
python gym/f110_gym/unittest/pyglet_test.py
```

### 주요 알고리즘

#### Single-Track Dynamic Model (STD)
7차원 상태 벡터를 사용한 차량 동역학:
- [x, y, steering_angle, velocity, yaw_angle, yaw_rate, slip_angle]
- 타이어 슬립과 비선형 동역학 고려

#### GJK 충돌 감지
- Gilbert-Johnson-Keerthi 알고리즘
- 볼록 다각형 간 효율적인 충돌 감지
- O(n) 시간 복잡도

#### 거리 변환 기반 LIDAR
- scipy.ndimage.edt를 사용한 거리 맵 생성
- 레이캐스팅으로 정확한 거리 측정
- 다른 차량에 의한 가림 효과 처리

### 기여하기

기여를 환영합니다! 이슈나 풀 리퀘스트를 GitHub 저장소에 제출해주세요.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 문서

자세한 문서는 `docs/` 디렉토리에서 확인할 수 있습니다:

- [기본 사용법](docs/basic_usage.rst)
- [고급 사용법 및 커스터마이징](docs/customized_usage.rst)
- [설치 가이드](docs/installation.rst)
- [API 문서](docs/api/)

또는 온라인 문서를 참조하세요: [F1TENTH Documentation](https://f1tenth.org)

## 인용

이 시뮬레이터를 연구에 사용하는 경우, 다음을 인용해주세요:

```bibtex
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

- 웹사이트: [https://f1tenth.org](https://f1tenth.org)
- GitHub: [https://github.com/f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)
- 이슈 리포트: [GitHub Issues](https://github.com/f1tenth/f1tenth_gym/issues)

---

**Note:** 이 버전은 ROS2 Humble과의 호환성을 위해 Gymnasium API로 업데이트되었습니다. 원본 gym 라이브러리를 사용하는 레거시 코드와는 API가 다를 수 있습니다. 자세한 내용은 [ROS2 Humble 호환성](#ros2-humble-호환성) 섹션을 참조하세요.
