"""
Improved Reward Functions for F1TENTH Racing

Ported from Isaac Lab's advanced reward shaping:
- Track progress tracking (centerline projection)
- Consecutive collision detection
- Speed rewards and slow penalties
- Exponential danger penalties (wall proximity)
- Stuck detection

Author: Ported from Isaac Lab f1tenth_env.py
"""

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces


class F110_ImprovedReward(gym.Wrapper):
    """
    Advanced reward wrapper for F1TENTH with improved reward shaping.

    Features:
    - Track progress tracking via centerline projection
    - Speed-based forward rewards
    - Exponential danger penalties for wall proximity
    - Consecutive LiDAR-based collision detection
    - Slow speed penalties to prevent stopping
    """

    def __init__(self, env, centerline_path=None, debug_mode=False):
        super().__init__(env)

        self.debug_mode = debug_mode

        # Load centerline waypoints for progress tracking
        if centerline_path is None:
            # Default to current map's centerline
            centerline_path = f"./f1tenth_racetracks/{env.map_name}/{env.map_name}_centerline.csv"

        self.centerline_waypoints = np.genfromtxt(centerline_path, delimiter=',')[:, :2]  # Only X, Y
        self._init_centerline()

        # Track progress state
        self.current_waypoint_idx = 0
        self.previous_progress = 0.0
        self.total_distance = 0.0
        self.lap_count = 0
        self.previous_pos = np.zeros(2)

        # Stuck detection
        self.stuck_check_interval = 120  # steps (2 seconds at 60Hz)
        self.stuck_threshold = 0.1  # meters (10cm in 2 seconds = avg 0.05 m/s)
        self.stuck_initial_delay = 300  # steps (5 seconds)
        self.last_check_pos = np.zeros(2)
        self.steps_since_last_check = 0

        # Episode statistics
        self.episode_step = 0
        self.episode_reward_sum = 0.0
        self.episode_forward_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_danger_sum = 0.0

        # Store LiDAR for collision detection
        self.lidar_distances = None

        print(f"[ImprovedReward] Initialized with centerline: {centerline_path}")
        print(f"  - Track length: {self.track_length:.2f}m")
        print(f"  - Num waypoints: {len(self.centerline)}")

    def _init_centerline(self):
        """
        Initialize centerline and interpolate waypoints for smooth progress tracking.
        """
        waypoints = self.centerline_waypoints

        # Linear interpolation between waypoints (100 points per segment = 1cm resolution)
        interpolated_points = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            for j in range(100):
                t = j / 100.0
                point = start + t * (end - start)
                interpolated_points.append(point)

        # Convert to numpy array
        self.centerline = np.array(interpolated_points, dtype=np.float32)

        # Compute cumulative distances along centerline
        distances = np.linalg.norm(self.centerline[1:] - self.centerline[:-1], axis=1)
        self.centerline_cumulative_dist = np.concatenate([
            np.zeros(1),
            np.cumsum(distances)
        ])

        # Total track length
        self.track_length = self.centerline_cumulative_dist[-1]

    def _get_track_progress(self, pos: np.ndarray) -> tuple:
        """
        Calculate track progress distance via segment projection.

        Projects vehicle position onto nearest centerline segment to get
        continuous progress measurement (handles car being off-centerline).

        Args:
            pos: vehicle XY position (2,)

        Returns:
            progress: track progress distance [0, track_length]
            progress_delta: change in progress since last step
        """
        # Segment projection method
        num_segments = len(self.centerline) - 1

        segment_starts = self.centerline[:-1]  # (num_segments, 2)
        segment_ends = self.centerline[1:]     # (num_segments, 2)
        segment_vectors = segment_ends - segment_starts
        segment_len_sq = np.sum(segment_vectors ** 2, axis=1)

        # Vector from segment start to vehicle
        to_vehicle = pos[np.newaxis, :] - segment_starts  # (num_segments, 2)

        # Projection ratio t = (to_vehicle · segment_vector) / ||segment_vector||^2
        dot_product = np.sum(to_vehicle * segment_vectors, axis=1)
        t = dot_product / (segment_len_sq + 1e-8)
        t = np.clip(t, 0.0, 1.0)  # Clamp to segment

        # Projected points
        projected_points = segment_starts + t[:, np.newaxis] * segment_vectors

        # Distance to projections
        dist_to_proj = np.linalg.norm(pos[np.newaxis, :] - projected_points, axis=1)

        # Find closest segment
        closest_seg = np.argmin(dist_to_proj)
        t_closest = t[closest_seg]

        # Progress distance = segment start distance + (t × segment length)
        seg_start_dist = self.centerline_cumulative_dist[closest_seg]
        seg_length = np.sqrt(segment_len_sq[closest_seg])
        progress = seg_start_dist + t_closest * seg_length

        # Detect finish line crossing (90% → 10% transition)
        finish_line_crossed = (self.previous_progress > self.track_length * 0.9) and \
                              (progress < self.track_length * 0.1)

        if finish_line_crossed:
            self.lap_count += 1

        # Absolute progress (continuous across laps)
        absolute_progress = progress + self.lap_count * self.track_length

        # Progress delta
        progress_delta = absolute_progress - self.total_distance

        # Update state
        self.total_distance = absolute_progress
        self.current_waypoint_idx = closest_seg
        self.previous_progress = progress

        return progress, progress_delta

    def _get_centerline_direction(self) -> np.ndarray:
        """
        Get centerline direction vector at current position.

        Returns:
            direction: normalized XY direction vector (2,)
        """
        segment_idx = self.current_waypoint_idx
        segment_start = self.centerline[segment_idx]
        segment_end = self.centerline[segment_idx + 1]

        direction = segment_end - segment_start
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        return direction

    def _detect_collision_consecutive(
        self,
        lidar_distances: np.ndarray,
        threshold: float = 0.15,
        consecutive_count: int = 10
    ) -> bool:
        """
        Detect collision using consecutive LiDAR points below threshold.

        More robust than min-distance: real collisions involve multiple
        adjacent rays detecting a close surface.

        Args:
            lidar_distances: LiDAR distance measurements (1080,)
            threshold: collision distance threshold (meters)
            consecutive_count: number of consecutive points required

        Returns:
            collision_detected: boolean
        """
        if lidar_distances is None:
            return False

        # Convert to torch for 1D convolution
        close_mask = (lidar_distances < threshold).astype(np.float32)
        close_mask_torch = torch.from_numpy(close_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, 1080)

        # Convolution kernel: [1, 1, ..., 1] of length consecutive_count
        kernel = torch.ones(1, 1, consecutive_count)

        # Convolve: output shows sum of consecutive values
        conv_result = F.conv1d(close_mask_torch, kernel, padding=consecutive_count // 2)

        # If any position has sum >= consecutive_count, collision detected
        has_consecutive = (conv_result >= consecutive_count).any().item()

        return has_consecutive

    def step(self, action):
        """
        Step with improved reward shaping.
        """
        # Original step
        observation, _, done, info = self.env.step(action)

        self.episode_step += 1
        self.steps_since_last_check += 1

        # Extract state
        pos_xy = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        vel_xy = np.array([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        lidar_scan = observation['scans'][0]
        self.lidar_distances = lidar_scan

        # --- Reward Calculation ---

        # 1) Track progress reward
        _, progress_delta = self._get_track_progress(pos_xy)
        reward_forward = np.clip(progress_delta, 0.0, None) * 18.0  # 18 points per meter

        # 2) Speed reward (forward direction)
        forward_direction = self._get_centerline_direction()
        forward_speed = np.sum(vel_xy * forward_direction)
        reward_speed = np.clip(forward_speed / 5.0, 0.0, 1.0) * 1.2  # Normalize by max speed

        # 3) Slow speed penalty (inverse function)
        TARGET_SPEED = 1.0  # m/s
        PENALTY_SCALE = 1.0
        MIN_SPEED_RATIO = 0.05  # Prevent explosion

        speed_ratio = np.clip(forward_speed / TARGET_SPEED, MIN_SPEED_RATIO, 10.0)
        reward_survival = -PENALTY_SCALE / speed_ratio + PENALTY_SCALE
        reward_survival = np.clip(reward_survival, -10.0, 1.0)

        # 4) Danger penalty (exponential wall proximity)
        min_distance = np.min(lidar_scan)

        WARNING_DISTANCE = 0.25  # 25cm
        PENALTY_SCALE_DANGER = 4.0
        EXP_STEEPNESS = 8.0

        proximity = np.clip(WARNING_DISTANCE - min_distance, 0.0, None)
        danger_penalty = -PENALTY_SCALE_DANGER * (np.exp(EXP_STEEPNESS * proximity) - 1.0)
        danger_penalty = np.clip(danger_penalty, -10.0, 0.0)

        # Total reward
        reward = reward_forward + reward_speed + reward_survival + danger_penalty

        # --- Termination Conditions ---

        # Collision detection (with grace period)
        grace_period = 5  # steps
        in_grace_period = self.episode_step < grace_period

        collision_detected = self._detect_collision_consecutive(
            lidar_scan,
            threshold=0.15,
            consecutive_count=10
        )

        if collision_detected and not in_grace_period:
            done = True
            if self.debug_mode:
                print(f"[Episode End] COLLISION at step {self.episode_step}")

        # Stuck detection (every stuck_check_interval steps)
        if self.steps_since_last_check >= self.stuck_check_interval:
            if self.episode_step > self.stuck_initial_delay:
                movement = np.linalg.norm(pos_xy - self.last_check_pos)
                if movement < self.stuck_threshold:
                    done = True
                    if self.debug_mode:
                        print(f"[Episode End] STUCK at step {self.episode_step}")

            self.last_check_pos = pos_xy.copy()
            self.steps_since_last_check = 0

        # Original collision flag (backward compatibility)
        if observation['collisions'][0]:
            done = True

        # --- Episode Statistics ---
        self.episode_reward_sum += reward
        self.episode_forward_sum += reward_forward
        self.episode_speed_sum += reward_speed
        self.episode_danger_sum += danger_penalty

        # Debug logging at episode end
        if done and self.debug_mode:
            avg_reward = self.episode_reward_sum / max(self.episode_step, 1)
            print(f"[Episode Summary] Steps: {self.episode_step}, Total Distance: {self.total_distance:.2f}m")
            print(f"  Rewards: Total={self.episode_reward_sum:.2f}, Avg={avg_reward:.3f}")
            print(f"  Forward: {self.episode_forward_sum:.2f}, Speed: {self.episode_speed_sum:.2f}, Danger: {self.episode_danger_sum:.2f}")

        # Update previous position
        self.previous_pos = pos_xy.copy()

        return observation['scans'][0], reward, done, info

    def reset(self, **kwargs):
        """
        Reset environment and internal state.
        """
        observation = self.env.reset(**kwargs)

        # Reset progress tracking
        pos_xy = np.array([observation['poses_x'][0], observation['poses_y'][0]])

        # Find initial progress
        _, _ = self._get_track_progress(pos_xy)
        self.previous_pos = pos_xy.copy()
        self.lap_count = 0

        # Reset stuck detection
        self.last_check_pos = pos_xy.copy()
        self.steps_since_last_check = -self.stuck_initial_delay  # Delay first check

        # Reset episode stats
        self.episode_step = 0
        self.episode_reward_sum = 0.0
        self.episode_forward_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_danger_sum = 0.0

        return observation['scans'][0]

    def seed(self, seed):
        self.env.seed(seed)
