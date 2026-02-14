import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrainTrafficEnv(gym.Env):
    def __init__(self):
        super(TrainTrafficEnv, self).__init__()
        # Actions: 0=Decelerate, 1=Maintain, 2=Accelerate
        self.action_space = spaces.Discrete(3)
        # State: [Position, Speed, Distance to lead train]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.state = np.array([0, 15, 100], dtype=np.float32) # Start state
        return self.state, {}

    def step(self, action):
        pos, speed, dist = self.state
        
        # AI Logic for Speed
        if action == 0: speed -= 1
        elif action == 2: speed += 1
        speed = np.clip(speed, 5, 40) # Speed limits (m/s)

        pos += speed
        dist -= 1 # Simplified gap closing
        
        # Reward Logic
        reward = (speed * 0.5) 
        if dist < 20: reward -= 50
        if dist > 20 and dist < 40: reward += 20

        done = pos >= 1000
        self.state = np.array([pos % 1000, speed, dist if dist > 0 else 100], dtype=np.float32)
        return self.state, reward, done, False, {}
