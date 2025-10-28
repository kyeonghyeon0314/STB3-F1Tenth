"""
SAC + CNN 정책 설정을 위한 통합 테스트 (개선판)

이 스크립트는 F1TENTH 환경, 커스텀 래퍼, 그리고 Stable Baselines 3 SAC 모델의
통합을 검증합니다. gym/gymnasium 호환성 문제와 경로 문제를 해결하도록
개선되었습니다.

테스트 항목:
1. 주요 컴포넌트 임포트
2. Gymnasium 호환성을 포함한 환경 생성 및 래핑
3. CNN 정책의 특징 추출기
4. SAC 모델 생성, 예측, 및 짧은 학습
5. 개선된 보상 함수 로직
"""

import gymnasium as gym
import f110_gym  # F1TENTH 환경 등록
import numpy as np
import torch
import os
import traceback

# --- 경로 설정 ---
# 스크립트의 위치를 기준으로 상대 경로를 사용하여 이식성을 높입니다.
try:
    # __file__은 스크립트로 실행될 때 정의됩니다.
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 대화형 환경(예: Jupyter)에서 실행될 경우를 대비합니다.
    ROOT_DIR = os.getcwd()

MAP_NAME = "underground"
RACETRACK_PATH = os.path.join(ROOT_DIR, "f1tenth_racetracks", MAP_NAME)
MAP_PATH = os.path.join(RACETRACK_PATH, f"{MAP_NAME}_map")
CENTERLINE_PATH = os.path.join(RACETRACK_PATH, f"{MAP_NAME}_centerline.csv")

# --- 컴포넌트 임포트 ---
# 임포트 실패 시 빠른 실패를 위해 테스트 함수 외부에서 임포트합니다.
try:
    from code.wrappers import F110_Wrapped
    from code.improved_rewards import F110_ImprovedReward
    from code.cnn_policy import CNNSACPolicy, LidarFeatureExtractor
    from stable_baselines3 import SAC
    print("✓ 모든 주요 컴포넌트 임포트 성공")
except ImportError as e:
    print(f"✗ 치명적 오류: 컴포넌트 임포트 실패: {e}")
    exit(1)


def print_test_header(name):
    print("\n" + "=" * 60)
    print(f"테스트: {name}")
    print("=" * 60)


def create_env(debug_rewards=False):
    """테스트를 위한 F1TENTH 환경을 생성하고 래핑하는 헬퍼 함수."""
    # 1. 기본 F110 환경 생성
    #    gym.make는 gymnasium의 일부이며, 오래된 gym 환경과의 호환성을 처리합니다.
    env = gym.make("f110-v0",
                   map=MAP_PATH,
                   map_ext=".png",
                   num_agents=1)
    print("  ✓ 기본 F110 환경 생성됨")

    # 2. F110_Wrapped로 래핑 (관측/행동 공간 처리)
    env = F110_Wrapped(env)
    print("  ✓ F110_Wrapped 적용됨")

    # 3. F110_ImprovedReward로 래핑 (보상 형성)
    #    중심선 파일이 존재하고 비어있지 않은지 명시적으로 확인합니다.
    assert os.path.exists(CENTERLINE_PATH), f"중심선 파일을 찾을 수 없습니다: {CENTERLINE_PATH}"
    assert os.path.getsize(CENTERLINE_PATH) > 0, f"중심선 파일이 비어있습니다: {CENTERLINE_PATH}"
    
    env = F110_ImprovedReward(env, centerline_path=CENTERLINE_PATH, debug_mode=debug_rewards)
    print("  ✓ F110_ImprovedReward 적용됨")
    
    return env


def test_environment_setup():
    """테스트 1: 환경 생성 및 래핑, reset/step 기능 테스트"""
    print_test_header("환경 설정 및 기본 API")
    try:
        env = create_env(debug_rewards=True)

        # 리셋 테스트: gymnasium API는 obs, info를 반환합니다.
        # 커스텀 래퍼가 이를 올바르게 처리해야 합니다.
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        print(f"  ✓ 환경 리셋 성공, obs 모양: {obs.shape}")
        assert isinstance(obs, np.ndarray), "관측값은 numpy 배열이어야 합니다."
        assert obs.shape == (1080,), f"관측값 모양이 예상과 다릅니다: {obs.shape}"

        # 스텝 테스트
        action = env.action_space.sample()
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_result
        print(f"  ✓ 환경 스텝 성공")
        print(f"    - 관측 모양: {obs.shape}")
        print(f"    - 보상: {reward:.3f}")
        print(f"    - 완료: {done}")
        
        env.close()
        print("✓ 환경 테스트 통과!")
        return True

    except Exception as e:
        print(f"✗ 환경 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_cnn_policy():
    """테스트 2: CNN 정책 특징 추출기 테스트"""
    print_test_header("CNN 정책 특징 추출기")
    try:
        # 특징 추출기 테스트를 위한 더미 관측 공간
        from gymnasium import spaces
        dummy_obs_space = spaces.Box(low=0.0, high=30.0, shape=(1080,), dtype=np.float32)
        
        feature_extractor = LidarFeatureExtractor(dummy_obs_space, features_dim=64)
        print("  ✓ LidarFeatureExtractor 생성됨")

        # 순전파 테스트 (4개 관측값 배치)
        dummy_obs_tensor = torch.randn(4, 1080)
        features = feature_extractor(dummy_obs_tensor)
        print(f"  ✓ 특징 추출 성공, 출력 모양: {features.shape}")
        
        assert features.shape == (4, 64), f"특징 모양이 예상과 다릅니다: {features.shape}"
        
        print("✓ CNN 정책 테스트 통과!")
        return True

    except Exception as e:
        print(f"✗ CNN 정책 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_sac_model_creation_and_use():
    """테스트 3: SAC 모델 생성, 예측 및 학습 테스트"""
    print_test_header("SAC 모델 생성 및 사용")
    try:
        env = create_env()
        print("  ✓ 테스트용 환경 생성됨")

        model = SAC(
            policy=CNNSACPolicy,
            env=env,
            verbose=0, # 테스트 중에는 로그 최소화
            device='cpu'
        )
        print("  ✓ CNN 정책으로 SAC 모델 생성됨")

        # 예측 테스트
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result
        action, _ = model.predict(obs, deterministic=True)
        print(f"  ✓ 모델 예측 성공, 행동: {action}")
        assert env.action_space.contains(action), "모델의 행동이 행동 공간 내에 있어야 합니다."

        # 짧은 학습 테스트
        model.learn(total_timesteps=10)
        print("  ✓ 짧은 훈련 실행 성공")

        env.close()
        print("✓ SAC 모델 테스트 통과!")
        return True

    except Exception as e:
        print(f"✗ SAC 모델 테스트 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """모든 통합 테스트 실행"""
    print("\n" + "=" * 60)
    print("SAC + CNN 정책을 위한 통합 테스트 스위트 (개선판)")
    print("=" * 60)

    tests = {
        "Environment Setup": test_environment_setup,
        "CNN Policy": test_cnn_policy,
        "SAC Model": test_sac_model_creation_and_use,
    }

    results = {}
    all_passed = True

    for test_name, test_func in tests.items():
        passed = test_func()
        results[test_name] = passed
        if not passed:
            all_passed = False

    # 요약
    print("\n" + "=" * 60)
    print("테스트 요약")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ 통과" if passed else "✗ 실패"
        print(f"- {test_name:30s}: {status}")
    
    print("=" * 60)
    if all_passed:
        print("🎉 모든 테스트 통과! 훈련 준비 완료.")
    else:
        print("🔥 일부 테스트 실패. 위에 출력된 오류를 확인하고 수정하십시오.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    # 테스트 실패 시 0이 아닌 종료 코드를 반환하여 CI/CD 파이프라인 등에서 실패를 감지할 수 있도록 합니다.
    exit(0 if success else 1)
