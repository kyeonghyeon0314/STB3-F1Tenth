"""
Custom 1D CNN Feature Extractor for Stable Baselines 3

This module provides a `BaseFeaturesExtractor` subclass that uses a 1D 
Convolutional Neural Network to process LiDAR scan data. This is designed to 
be used with policies like MlpPolicy in Stable Baselines 3.

The architecture is inspired by TinyLidarNet for F1TENTH racing.
Original architecture: https://arxiv.org/html/2410.07447v1
"""

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarFeatureExtractor(BaseFeaturesExtractor):
    """
    1D CNN for LiDAR feature extraction.

    Compresses a 1080-dimensional LiDAR scan into a lower-dimensional feature vector
    by learning spatial patterns (walls, corners, obstacles).

    Architecture:
        Conv1d(1, 32, k=5, s=2)  → ReLU # 1080 -> 540
        Conv1d(32, 64, k=3, s=2) → ReLU # 540  -> 270
        Conv1d(64, 64, k=3, s=2) → ReLU # 270  -> 135
        GlobalAveragePooling1D()      # 64x135 -> 64
    
    :param observation_space: The Gym observation space.
    :param features_dim: Number of features to extract.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # Check that the observation space is a 1D Box with 1080 entries
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
        
        # Output is 64 features
        print(f"[LidarFeatureExtractor] Initialized: 1080 -> CNN -> {features_dim} features")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        :param observations: LiDAR scans of shape (batch_size, 1080)
        :return: Extracted features of shape (batch_size, 64)
        """
        # Add a channel dimension: (batch_size, 1080) -> (batch_size, 1, 1080)
        cnn_input = observations.unsqueeze(1)
        
        # Pass through convolutional layers
        features = self.cnn(cnn_input) # -> (batch_size, 64, 135)
        
        # Apply global average pooling
        # Averages across the spatial dimension (135)
        pooled_features = torch.nn.functional.adaptive_avg_pool1d(features, 1) # -> (batch_size, 64, 1)
        
        # Remove the last dimension
        return pooled_features.squeeze(-1) # -> (batch_size, 64)

