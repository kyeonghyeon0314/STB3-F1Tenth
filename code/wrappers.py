# MIT License

# Copyright (c) 2020 FT Autonomous Team One

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
import numpy as np

from gym import spaces
from pathlib import Path

from code.random_trackgen import create_track, convert_track

mapno = ["Underground"]

globwaypoints = np.genfromtxt(f"./f1tenth_racetracks/{mapno[0]}/{mapno[0]}_centerline.csv", delimiter=',')

def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    Wrapper for the F1Tenth Gym environment to be used with Stable Baselines 3.
    It returns only LiDAR scans as observations and handles action normalization.
    """

    def __init__(self, env):
        super().__init__(env)

        # Normalized action space for steering and speed, range [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space is the LiDAR scan data (raw distances)
        self.lidar_max = 30.0  # Max range of the LiDAR
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.lidar_max,
            shape=(1080,),
            dtype=np.float32
        )

        # Store steering and speed ranges for action conversion
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']

        # Store car dimensions and track info for reset logic
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 3.2
        self.start_radius = (self.track_width / 2) - ((self.car_length + self.car_width) / 2)

        self.step_count = 0
        self.max_theta = 100 # Threshold for maximum car angle to prevent spinning

    def step(self, action):
        # Convert normalized actions back to simulator's expected range
        converted_action = self.un_normalise_actions(action)
        observation, _, done, info = self.env.step(np.array([converted_action]))

        self.step_count += 1

        # --- Basic Reward Function ---
        # Positive reward for speed, negative reward for collision
        vel_magnitude = np.linalg.norm([
            observation['linear_vels_x'][0],
            observation['linear_vels_y'][0]
        ])
        
        if observation['collisions'][0]:
            reward = -1.0 # Collision penalty
        else:
            reward = vel_magnitude # Reward for forward speed
        
        # End episode if car spins out
        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True

        return observation['scans'][0].astype(np.float32), reward, bool(done), info

    def reset(self, start_xy=None, direction=None):
        if start_xy is None:
            start_xy = np.zeros(2)
        if direction is None:
            direction = np.random.uniform(0, 2 * np.pi)

        # Get a random starting position on the track
        slope = np.tan(direction + np.pi / 2)
        magnitude = np.sqrt(1 + np.power(slope, 2))
        rand_offset = np.random.uniform(-1, 1) * self.start_radius
        x, y = start_xy + rand_offset * np.array([1, slope]) / magnitude

        # Point car in a random forward direction
        theta = np.random.uniform(-np.pi / 4, np.pi / 4) + direction
        
        # Reset car pose
        observation, _, _, _ = self.env.reset(np.array([[x, y, theta]]))
        
        return observation['scans'][0].astype(np.float32)

    def un_normalise_actions(self, actions: np.ndarray) -> np.ndarray:
        """Converts actions from [-1, 1] range to steering/speed range."""
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed])

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed: int):
        self.env.seed(seed)
        np.random.seed(seed)
        print(f"F110_Wrapped seeded with {seed}")


class RandomMap(gym.Wrapper):
    """
    Generates random maps at chosen intervals, when resetting car,
    and positions car at random point around new track
    """

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # create map
            for _ in range(self.MAX_CREATE_ATTEMPTS):
                try:
                    track, track_int, track_ext = create_track()
                    convert_track(track,
                                  track_int,
                                  track_ext,
                                  self.current_seed)
                    break
                except Exception:
                    print(
                        f"Random generator [{self.current_seed}] failed, trying again...")
            # update map
            self.update_map(f"./maps/map{self.current_seed}", ".png")
            # store waypoints
            self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",
                                           delimiter=',')
        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]
        print(start_xy)
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass

class RandomF1TenthMap(gym.Wrapper):
    """
    Places the car in a random map from F1Tenth
    """

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # update map
            randmap = mapno[np.random.randint(low=0, high=22)]
            #self.update_map(f"./maps/map{self.current_seed}", ".png")
            self.update_map(f"./f1tenth_racetracks/{randmap}/{randmap}_map", ".png")
            # store waypoints
            #self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",delimiter=',')
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
            globwaypoints = self.waypoints

        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]  #len = 4
        start_xy = start_xy[:2]
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass


class ThrottleMaxSpeedReward(gym.RewardWrapper):
    """
    Slowly increase maximum reward for going fast, so that car learns
    to drive well before trying to improve speed
    """

    def __init__(self, env, start_step, end_step, start_max_reward, end_max_reward=None):
        super().__init__(env)
        # initialise step boundaries
        self.end_step = end_step
        self.start_step = start_step
        self.start_max_reward = start_max_reward
        # set finishing maximum reward to be maximum possible speed by default
        self.end_max_reward = self.v_max if end_max_reward is None else end_max_reward

        # calculate slope for reward changing over time (steps)
        self.reward_slope = (self.end_max_reward - self.start_max_reward) / (self.end_step - self.start_step)

    def reward(self, reward):
        # maximum reward is start_max_reward
        if self.step_count < self.start_step:
            return min(reward, self.start_max_reward)
        # maximum reward is end_max_reward
        elif self.step_count > self.end_step:
            return min(reward, self.end_max_reward)
        # otherwise, proportional reward between two step endpoints
        else:
            return min(reward, self.start_max_reward + (self.step_count - self.start_step) * self.reward_slope)
