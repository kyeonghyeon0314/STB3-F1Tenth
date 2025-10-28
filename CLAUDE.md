# Gemini 코드 어시스턴트 컨텍스트: F1Tenth-RL 프로젝트

## 프로젝트 개요

이 프로젝트는 F1/10 스케일의 자율 주행 레이싱 카를 위한 강화학습(RL) 에이전트를 훈련하는 것을 목표로 합니다. 주 환경으로는 ROS 기반 시뮬레이터인 `f1tenth_gym`을 사용합니다.

프로젝트의 핵심은 **Soft Actor-Critic (SAC)** 알고리즘을 사용하는 고급 에이전트입니다. SAC는 연속적인 제어 작업에서 샘플 효율성이 높기로 알려진 off-policy 방식의 알고리즘입니다. 에이전트의 정책(policy)은 1D LiDAR 스캔 데이터(1080개 포인트)를 직접 처리하여 벽이나 코너와 같은 공간적 특징을 추출하는 커스텀 **Convolutional Neural Network (CNN)** 아키텍처(`TinyLidarNet`)를 사용합니다. 이는 기존의 PPO+MLP 모델에 비해 크게 향상된 부분입니다.

훈련 과정에는 NVIDIA의 Isaac Lab에서 가져온 정교한 보상 함수가 통합되어 있으며, 다음 요소들을 장려합니다:
- **트랙 진행:** 트랙 중앙선을 따라 진행한 거리에 대해 높은 보상.
- **속도:** 높은 전진 속도를 유지하는 것에 대한 긍정적 보상.
- **안전:** 벽에 너무 가까워지거나(`Danger Penalty`) 너무 느리게 움직일 때 패널티 부과.

이 프로젝트는 Docker 컨테이너 내에서 실행되도록 구성되어 있으며, 핵심 강화학습 라이브러리로 `stable-baselines3`를 사용합니다.

## 주요 파일

- `training_sac_cnn.py`: SAC+CNN 에이전트를 훈련하기 위한 메인 스크립트.
- `evaluating.py`: 훈련된 에이전트의 성능을 평가하기 위한 스크립트.
- `test_integration.py`: 장시간의 훈련을 시작하기 전, 모든 구성 요소(환경, 정책, 보상, 모델)가 올바르게 설정되었는지 확인하는 필수 유틸리티.
- `code/cnn_policy.py`: 커스텀 `CNNSACPolicy`와 `LidarFeatureExtractor` CNN 모델을 정의.
- `code/wrappers.py`: 관측값 처리, 무작위 맵 생성 등 다양한 Gym 환경 래퍼(wrapper)를 포함.
- `code/improved_rewards.py`: Isaac Lab 스타일의 상세한 보상 함수를 구현.
- `f1tenth_gym_ros/`: F1Tenth Gym 환경을 포함하는 Git 서브모듈.
- `Dockerfile`: 필요한 모든 의존성(PyTorch, Stable Baselines 3, ROS)을 포함하는 Docker 이미지를 정의.

## 개발 워크플로우

모든 명령어는 프로젝트의 Docker 컨테이너 내에서 실행해야 합니다.

### 1. 초기 설정 및 검증

먼저, 통합 테스트 스위트를 실행하여 환경이 올바르게 설정되었는지 확인합니다. 이 테스트는 임포트, 환경 래핑, 정책 생성 및 모델 기능이 정상적으로 작동하는지 확인합니다.

```bash
python3 test_integration.py
```
성공적으로 실행되면 `✓ ALL TESTS PASSED! Ready to train.` 메시지가 출력됩니다.

### 2. 에이전트 훈련

주요 훈련 스크립트는 `training_sac_cnn.py`입니다.

**새로운 훈련 시작:**
```bash
python3 training_sac_cnn.py
```

**최신 체크포인트에서 훈련 재개:**
```bash
python3 training_sac_cnn.py --load latest
```

**로깅을 위해 Weights & Biases 사용:**
```bash
python3 training_sac_cnn.py --wandb
```

**보상 함수에 대한 상세 디버그 로그 활성화:**
```bash
python3 training_sac_cnn.py --debug
```

### 3. 훈련 모니터링

훈련 진행 상황은 TensorBoard를 통해 모니터링할 수 있습니다.

```bash
tensorboard --logdir=./sac_cnn_tensorboard
```

`--wandb` 플래그를 사용하면, 결과는 Weights & Biases의 `f1tenth-sac-cnn` 프로젝트에서도 확인할 수 있습니다.

### 4. 에이전트 평가

훈련된 모델을 평가하려면 `evaluating.py` 스크립트를 사용합니다. (참고: 정확한 인자는 파일을 직접 읽어 확인해야 할 수 있지만, 저장된 모델 `.zip` 파일의 경로가 필요할 것입니다).

### 빌드 및 실행 명령어

전체 환경은 Docker를 통해 관리됩니다.

**컨테이너 빌드 및 시작 (컨테이너 외부에서 실행):**
```bash
# Linux 환경
source scripts/docker-linux.sh

# Windows 환경
call scripts/docker-windows.bat
```

**실행 중인 컨테이너에 VS Code 연결:**
1.  `Remote - Containers` (`ms-vscode-remote.remote-containers`) 확장 프로그램을 설치합니다.
2.  명령 팔레트(`Ctrl+Shift+P`)를 열고 `Remote-Containers: Attach to Running Container`를 실행합니다.
3.  `f110-rl-container`를 선택합니다.