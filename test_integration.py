"""
Integration test for SAC + CNN policy setup

Tests:
1. CNN policy imports correctly
2. Environment wrapping works
3. Model can be created and perform forward pass
4. Improved reward wrapper functions properly
"""

import gym
import numpy as np
import torch

from code.wrappers import F110_Wrapped
from code.improved_rewards import F110_ImprovedReward
from code.cnn_policy import CNNSACPolicy
from stable_baselines3 import SAC


def test_imports():
    """Test 1: Check if all imports work"""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)

    try:
        from code.cnn_policy import CNNSACPolicy, LidarFeatureExtractor
        print("✓ CNN policy imports successful")

        from code.improved_rewards import F110_ImprovedReward
        print("✓ Improved rewards imports successful")

        from stable_baselines3 import SAC
        print("✓ Stable Baselines 3 SAC imports successful")

        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment():
    """Test 2: Check if environment wrapping works"""
    print("\n" + "=" * 60)
    print("TEST 2: Testing environment wrapping...")
    print("=" * 60)

    try:
        # Create base environment
        env = gym.make("f110_gym:f110-v0",
                       map="./f1tenth_gym/examples/example_map",
                       map_ext=".png",
                       num_agents=1)
        print("✓ Base F110 environment created")

        # Wrap with F110_Wrapped
        env = F110_Wrapped(env)
        print("✓ F110_Wrapped applied")

        # Wrap with improved rewards
        env = F110_ImprovedReward(env, debug_mode=True)
        print("✓ F110_ImprovedReward applied")

        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")

        env.close()
        print("✓ Environment test successful!")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_policy():
    """Test 3: Check if CNN policy can be created and used"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing CNN policy...")
    print("=" * 60)

    try:
        # Create dummy environment for policy testing
        env = gym.make("f110_gym:f110-v0",
                       map="./f1tenth_gym/examples/example_map",
                       map_ext=".png",
                       num_agents=1)
        env = F110_Wrapped(env)
        print("✓ Test environment created")

        # Test feature extractor
        from code.cnn_policy import LidarFeatureExtractor
        feature_extractor = LidarFeatureExtractor(env.observation_space, features_dim=64)
        print("✓ LidarFeatureExtractor created")

        # Test forward pass
        dummy_obs = torch.randn(4, 1080)  # Batch of 4 observations
        features = feature_extractor(dummy_obs)
        print(f"✓ Feature extraction successful, output shape: {features.shape}")
        assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"

        env.close()
        print("✓ CNN policy test successful!")
        return True

    except Exception as e:
        print(f"✗ CNN policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sac_model():
    """Test 4: Check if SAC model can be created with CNN policy"""
    print("\n" + "=" * 60)
    print("TEST 4: Testing SAC model creation...")
    print("=" * 60)

    try:
        # Create environment
        env = gym.make("f110_gym:f110-v0",
                       map="./f1tenth_gym/examples/example_map",
                       map_ext=".png",
                       num_agents=1)
        env = F110_Wrapped(env)
        env = F110_ImprovedReward(env, debug_mode=False)
        print("✓ Environment created")

        # Create SAC model with CNN policy
        model = SAC(
            policy=CNNSACPolicy,
            env=env,
            learning_rate=3e-4,
            buffer_size=10000,  # Small buffer for testing
            learning_starts=100,
            batch_size=64,
            verbose=1,
            device='cpu'  # Use CPU for testing
        )
        print("✓ SAC model created with CNN policy")

        # Test prediction
        obs = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"✓ Model prediction successful, action: {action}")

        # Test learning (just a few steps)
        print("Testing short training run (10 steps)...")
        model.learn(total_timesteps=10, log_interval=None)
        print("✓ Short training run successful")

        env.close()
        print("✓ SAC model test successful!")
        return True

    except Exception as e:
        print(f"✗ SAC model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_functions():
    """Test 5: Check reward computation details"""
    print("\n" + "=" * 60)
    print("TEST 5: Testing reward functions...")
    print("=" * 60)

    try:
        env = gym.make("f110_gym:f110-v0",
                       map="./f1tenth_gym/examples/example_map",
                       map_ext=".png",
                       num_agents=1)
        env = F110_Wrapped(env)
        env = F110_ImprovedReward(env, debug_mode=True)

        obs = env.reset()
        print("✓ Environment reset")

        # Run a few steps and check rewards
        total_reward = 0.0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print(f"  Episode ended at step {i+1}")
                break

        print(f"✓ Ran 10 steps, total reward: {total_reward:.3f}")

        # Test collision detection
        print("\nTesting collision detection...")
        dummy_lidar = np.ones(1080) * 2.0  # All safe
        collision = env._detect_collision_consecutive(dummy_lidar, threshold=0.15, consecutive_count=10)
        print(f"  Safe LiDAR: collision={collision} (expected: False)")
        assert not collision, "False positive collision detection!"

        dummy_lidar = np.ones(1080) * 0.05  # All very close
        collision = env._detect_collision_consecutive(dummy_lidar, threshold=0.15, consecutive_count=10)
        print(f"  Collision LiDAR: collision={collision} (expected: True)")
        assert collision, "Collision not detected!"

        print("✓ Collision detection working correctly")

        env.close()
        print("✓ Reward functions test successful!")
        return True

    except Exception as e:
        print(f"✗ Reward functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUITE FOR SAC + CNN POLICY")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("CNN Policy", test_cnn_policy),
        ("SAC Model", test_sac_model),
        ("Reward Functions", test_reward_functions),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED! Ready to train.")
    else:
        print("✗ SOME TESTS FAILED. Please fix before training.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
