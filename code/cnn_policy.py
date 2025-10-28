"""
Stable Baselines 3를 위한 맞춤형 1D CNN 특징 추출기

이 모듈은 1D 컨볼루션 신경망을 사용하여 LiDAR 스캔 데이터를 처리하는
`BaseFeaturesExtractor` 서브클래스를 제공합니다. 이는 Stable Baselines 3의
MlpPolicy와 같은 정책과 함께 사용하도록 설계되었습니다.

이 아키텍처는 F1TENTH 레이싱을 위한 TinyLidarNet에서 영감을 받았습니다.
원본 아키텍처: https://arxiv.org/html/2410.07447v1
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarFeatureExtractor(BaseFeaturesExtractor):
    """
    LiDAR 특징 추출을 위한 1D CNN.

    1080차원 LiDAR 스캔을 공간적 패턴(벽, 코너, 장애물)을 학습하여
    저차원 특징 벡터로 압축합니다.

    아키텍처:
        Conv1d(1, 32, k=5, s=2)  → ReLU # 1080 -> 540
        Conv1d(32, 64, k=3, s=2) → ReLU # 540  -> 270
        Conv1d(64, 64, k=3, s=2) → ReLU # 270  -> 135
        GlobalAveragePooling1D()      # 64x135 -> 64
    
    :param observation_space: Gym 관측 공간.
    :param features_dim: 추출할 특징의 수.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # 관측 공간이 1080개 항목을 가진 1D Box인지 확인
        if not (len(observation_space.shape) == 1 and observation_space.shape[0] == 1080):
            raise ValueError(
                f"Expected observation space to be a 1D Box of shape (1080,), "
                f"but got {observation_space.shape}"
            )

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 출력은 64개의 특징
        print(f"[LidarFeatureExtractor] Initialized: 1080 -> CNN -> {features_dim} features")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        특징 추출기의 순전파.
        
        :param observations: (batch_size, 1080) 모양의 LiDAR 스캔
        :return: (batch_size, 64) 모양의 추출된 특징
        """
        # 채널 차원 추가: (batch_size, 1080) -> (batch_size, 1, 1080)
        cnn_input = observations.unsqueeze(1)
        
        # 컨볼루션 레이어를 통과
        features = self.cnn(cnn_input) # -> (batch_size, 64, 135)
        
        # 전역 평균 풀링 적용
        # 공간 차원(135)에 대해 평균 계산
        pooled_features = torch.nn.functional.adaptive_avg_pool1d(features, 1) # -> (batch_size, 64, 1)
        
        # 마지막 차원 제거
        return pooled_features.squeeze(-1) # -> (batch_size, 64)


# SAC 정책 기본 클래스 가져오기
from stable_baselines3.sac.policies import SACPolicy
from typing import Any, Dict, List, Optional, Type, Union, Callable


class CNNSACPolicy(SACPolicy):
    """
    LiDAR 특징 추출을 위해 1D CNN을 사용하는 맞춤형 SAC 정책.

    이 정책은 다음을 결합합니다:
    - LidarFeatureExtractor (1080 -> 64 CNN 특징)
    - 액터 네트워크 (정책): 64 -> 64 -> 64 -> 64 -> 행동 (평균, 로그 표준편차)
    - 크리틱 네트워크 (Q-함수): 64 + 행동 -> 128 -> 128 -> 64 -> Q-값

    액터는 가우시안 분포를 사용하여 연속적인 행동(조향, 속도)을 출력합니다.
    크리틱은 안정성을 위해 두 개의 Q-네트워크를 사용하여 상태-행동 쌍을 평가합니다.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        **kwargs
    ):
        """
        맞춤형 네트워크 아키텍처로 CNNSACPolicy를 초기화합니다.

        Args:
            observation_space: Gym 관측 공간 (모양이 (1080,)인 Box여야 함)
            action_space: Gym 행동 공간 (조향/속도를 위해 모양이 (2,)인 Box여야 함)
            lr_schedule: 학습률 스케줄 함수
            net_arch: 'pi' (액터)와 'qf' (크리틱) 키를 가진 네트워크 아키텍처 사전.
                     None이면 위에서 설명한 기본 아키텍처를 사용합니다.
            **kwargs: SACPolicy에 전달되는 추가 인수
        """
        # 기본 네트워크 아키텍처
        if net_arch is None:
            # 액터 (pi): 특징 벡터 (64) -> 64 -> 64 -> 64 -> 행동
            # 크리틱 (qf): 특징 벡터 (64) + 행동 (2) -> 128 -> 128 -> 64 -> Q-값
            net_arch = dict(
                pi=[64, 64, 64],      # ACTOR network layers
                qf=[128, 128, 64]     # CRITIC network layers
            )

        # 맞춤형 특징 추출기로 부모 SAC 정책 초기화
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            features_extractor_class=LidarFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            **kwargs
        )

        print("[CNNSACPolicy] Initialized SAC policy with CNN feature extractor")
        print(f"  - Feature extractor: LiDAR (1080) -> CNN -> 64 features")
        print(f"  - Actor network: {net_arch['pi']}")
        print(f"  - Critic network: {net_arch['qf']}")


# 정책 생성을 위한 편의 함수
def make_cnn_sac_policy(**kwargs):
    """
    맞춤형 kwargs로 CNNSACPolicy를 생성하는 팩토리 함수.

    사용법:
        policy = make_cnn_sac_policy(net_arch=dict(pi=[32, 32], qf=[64, 64]))
    """
    return CNNSACPolicy