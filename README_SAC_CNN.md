# F1TENTH SAC + CNN Policy Training

Isaac Lab의 고급 보상 함수와 CNN 아키텍처를 Stable Baselines 3 + f1tenth_gym에 이식한 버전입니다.

## 🎯 주요 개선사항

### 1. **CNN Policy (TinyLidarNet 아키텍처)**
- **기존**: MLP Policy (1080 → 64 → 64 → actions)
- **개선**: CNN + MLP Policy
  ```
  LiDAR (1080) → 1D CNN → 64 features → MLP → actions/Q-values
  ```
- **효과**: 공간적 패턴 학습 (벽, 코너, 장애물)

### 2. **SAC 알고리즘**
- **기존**: PPO (on-policy)
- **개선**: SAC (off-policy)
- **장점**:
  - 샘플 효율성 증가
  - 연속 제어에 최적화
  - 자동 엔트로피 튜닝

### 3. **고급 보상 함수**
Isaac Lab의 보상 설계 이식:

| 보상 요소 | 설명 | 가중치 |
|---------|------|--------|
| **Track Progress** | Centerline projection 기반 진행 거리 | +18 per meter |
| **Speed Reward** | 전진 속도 (track 방향) | +1.2 max |
| **Slow Penalty** | 너무 느린 속도 패널티 (inverse) | -10.0 max |
| **Danger Penalty** | 벽 근접 패널티 (exponential) | -10.0 max |

### 4. **개선된 충돌 감지**
- **기존**: 단순 최소 거리 체크
- **개선**: 연속적인 LiDAR 포인트 체크 (10개 연속)
- **효과**: False positive 감소

### 5. **Stuck Detection**
- 2초간 10cm 미만 이동 시 에피소드 종료
- 학습 초반 정체 방지

---

## 📁 새로 추가된 파일

```
F1Tenth-RL/
├── code/
│   ├── cnn_policy.py           # CNN feature extractor + SAC policy
│   └── improved_rewards.py     # Isaac Lab 보상 함수 이식
├── training_sac_cnn.py         # SAC + CNN 학습 스크립트
├── test_integration.py         # 통합 테스트 스크립트
├── training_ppo_backup.py      # 기존 PPO 스크립트 백업
└── README_SAC_CNN.md          # 이 문서
```

---

## 🚀 사용 방법

### 1. 통합 테스트 실행

먼저 모든 컴포넌트가 정상 작동하는지 확인:

```bash
python3 test_integration.py
```

**예상 출력**:
```
TEST SUMMARY
==============================================================
Imports             : ✓ PASSED
Environment         : ✓ PASSED
CNN Policy          : ✓ PASSED
SAC Model           : ✓ PASSED
Reward Functions    : ✓ PASSED
==============================================================
✓ ALL TESTS PASSED! Ready to train.
```

### 2. 학습 시작

#### 새로운 모델 학습
```bash
python3 training_sac_cnn.py
```

#### 체크포인트에서 이어서 학습
```bash
# 최신 모델 로드
python3 training_sac_cnn.py --load latest

# 특정 모델 로드
python3 training_sac_cnn.py --load sac-cnn-28-10-2025
```

#### WandB 로깅 사용
```bash
python3 training_sac_cnn.py --wandb
```

#### 디버그 모드 (에피소드 종료 시 상세 로그)
```bash
python3 training_sac_cnn.py --debug
```

#### 모든 옵션 조합
```bash
python3 training_sac_cnn.py --load latest --wandb --debug --save
```

### 3. 학습 모니터링

#### TensorBoard
```bash
tensorboard --logdir=./sac_cnn_tensorboard
```

#### WandB
프로젝트 이름: `f1tenth-sac-cnn`

---

## ⚙️ 하이퍼파라미터

### SAC 기본 설정 (training_sac_cnn.py)

```python
SAC(
    policy=CNNSACPolicy,
    learning_rate=3e-4,
    buffer_size=100000,       # Replay buffer 크기
    learning_starts=1000,     # 학습 시작 스텝
    batch_size=256,
    tau=0.005,                # Soft update coefficient
    gamma=0.99,               # Discount factor
    ent_coef='auto',          # 자동 엔트로피 튜닝
)
```

### CNN 아키텍처

**Feature Extractor**:
```
Conv1d(1→32, k=5, s=2) + ReLU    # 1080 → 540
Conv1d(32→64, k=3, s=2) + ReLU   # 540 → 270
Conv1d(64→64, k=3, s=2) + ReLU   # 270 → 135
Global Average Pooling           # 64×135 → 64
```

**Actor** (Policy):
```
64 → 64 → 64 → actions (2)
```

**Critic** (Q-function):
```
(64 + 2) → 128 → 128 → 64 → 1
```

### 보상 함수 파라미터 (improved_rewards.py)

튜닝이 필요하면 다음 파일 수정:
- `code/improved_rewards.py` → `step()` 메서드 내부

```python
# 진행 거리 가중치
reward_forward = progress_delta * 18.0  # 18점 per meter

# 속도 보상
reward_speed = (forward_speed / 5.0) * 1.2  # 5 m/s 기준

# 느린 속도 패널티
TARGET_SPEED = 1.0  # m/s
PENALTY_SCALE = 1.0

# 벽 근접 패널티
WARNING_DISTANCE = 0.25  # 25cm
PENALTY_SCALE_DANGER = 4.0
EXP_STEEPNESS = 8.0
```

---

## 🔧 트러블슈팅

### 문제: PyTorch 버전 충돌
```bash
pip install torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### 문제: f1tenth_gym import 오류
서브모듈이 제대로 초기화되지 않은 경우:
```bash
git submodule update --init --recursive
cd f1tenth_gym/gym
pip install -e .
```

### 문제: LiDAR shape mismatch
현재 설정은 1080개 LiDAR rays를 가정합니다. 다른 설정 사용 시:
- `code/cnn_policy.py` → `LidarFeatureExtractor.__init__()` 수정
- `code/wrappers.py` → observation_space 수정

### 문제: 학습이 너무 느림
1. `NUM_PROCESS` 증가 (training_sac_cnn.py)
   ```python
   NUM_PROCESS = 8  # CPU 코어 수에 맞게
   ```

2. GPU 사용
   ```python
   model = SAC(..., device='cuda')
   ```

---

## 📊 성능 비교 (예상)

| 메트릭 | PPO + MLP | SAC + CNN |
|--------|-----------|-----------|
| **샘플 효율성** | 1x | 2-3x |
| **최종 성능** | Baseline | +20-30% |
| **학습 안정성** | 중간 | 높음 |
| **벽 충돌** | 많음 | 적음 |

---

## 🎓 참고 자료

### Isaac Lab F1TENTH
- 원본 코드: [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- 논문: TinyLidarNet - [arXiv:2410.07447](https://arxiv.org/html/2410.07447v1)

### Stable Baselines 3
- 문서: [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- 예제: [Custom Policy](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)

### F1TENTH Gym
- 문서: [F1TENTH Gym](https://f1tenth-gym.readthedocs.io/)

---

## 📝 다음 단계

### 추가 개선 아이디어

1. **멀티 에이전트 학습**
   - 현재는 단일 에이전트만 지원
   - MAPPO/IPPO로 확장 가능

2. **도메인 랜덤화**
   - 트랙 마찰 계수 변화
   - LiDAR 노이즈 추가
   - 차량 파라미터 변화

3. **Privileged Learning**
   - Teacher network (full state)
   - Student network (LiDAR only)
   - Knowledge distillation

4. **모델 압축**
   - Quantization (INT8)
   - Pruning
   - Jetson 배포 최적화

---

## 🤝 기여

개선 사항이나 버그 발견 시:
1. Issue 생성
2. Pull Request 제출
3. 성능 비교 결과 공유

---

## 📄 라이선스

MIT License (원본 프로젝트 라이선스 유지)

---

**Happy Training! 🏎️💨**
