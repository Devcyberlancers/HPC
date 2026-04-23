import gymnasium as gym
from gymnasium import spaces
import numpy as np


class IrrigationEnv(gym.Env):

    def __init__(self):
        super(IrrigationEnv, self).__init__()

        # 🌱 STATE: [moisture, temp, humidity, N, P, K]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1, 50, 100, 100, 100, 100]),
            dtype=np.float32
        )

        # 💧 ACTION: irrigation (0 → 1)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 🔁 Random initial state
        self.state = np.array([
            np.random.uniform(0.1, 0.8),   # soil moisture
            np.random.uniform(20, 40),     # temperature
            np.random.uniform(30, 80),     # humidity
            np.random.uniform(20, 80),     # N
            np.random.uniform(20, 80),     # P
            np.random.uniform(20, 80),     # K
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):

        moisture, temp, humidity, N, P, K = self.state

        irrigation = float(action[0])

        # 🌱 Moisture dynamics
        moisture = moisture + irrigation * 0.4 - temp * 0.01
        moisture = np.clip(moisture, 0, 1)

        # 🎯 BALANCED REWARD SYSTEM (STABLE)
        reward = 0

        # ✅ Ideal moisture zone
        if 0.4 <= moisture <= 0.7:
            reward += 2

        # ❌ Too dry
        if moisture < 0.3:
            reward -= 2

        # ❌ Doing nothing when dry
        if moisture < 0.3 and irrigation < 0.3:
            reward -= 3

        # ❌ Overwatering
        if moisture > 0.85:
            reward -= 1.5

        # ✅ Good action when dry
        if moisture < 0.3 and irrigation > 0.5:
            reward += 2

        # 🔁 Update state
        self.state = np.array(
            [moisture, temp, humidity, N, P, K],
            dtype=np.float32
        )

        return self.state, reward, False, False, {}