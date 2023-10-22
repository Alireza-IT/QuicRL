import gymnasium as gym
import numpy as np
from gymnasium import spaces
import get_data
from aioquic.quic.recovery import QuicCongestionControl

quic_con = QuicCongestionControl()

class RL(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(5,2), dtype=np.float32)
        self.current_reward = 0.0
        self.prev_reward = 0.0
        self.reward = 0.0
        self.length = 60
        self.delta = 0.1
        self.reduction = QuicCongestionControl().reduction
        self.observation = None
        self.done = False

    def act_apply(self, action):
        self.reduction= self.reduction + (action * self.delta)

    def calculate_reward(self):
        average = np.mean(self.observation, axis=0)
        maximum = np.max(self.observation , axis = 0)
        reward = (average[0]/maximum[0]) - (average[1]/maximum[1])
        # print(f"this is current observation {self.observation} , throughput avg us {average}, maximum is {maximum}")
        # print(f"this is reward {reward}")
        return reward
        
    def step(self, action):
        self.length -=1
        self.act_apply(action)
        self.current_reward = self.calculate_reward()
        if self.current_reward > self.prev_reward : 
            self.reward += self.current_reward
            self.prev_reward = self.current_reward
        else:
            self.reward -= self.current_reward
            self.prev_reward = self.current_reward
        if self.length <=0:
            self.done = True
        else : 
            self.done : False
        info = {}
        return self.observation, self.current_reward, self.done, info

    def reset(self, seed=None, options=None): #during initialization it may called
        self.done = False
        self.observation = np.array(get_data.recieve_data())
        return self.observation

    # def render(self):
    #     ...

    # def close(self):
    #     ...

# env = RL()
# sample = env.observation_space.sample()
# print(sample)
# print(sample[1])
# print(sample[1])˜
# print(env.reset())
# env.act_apply(1)
# env.act_apply(0)
# env.calculate_reward()
# print(env.reset())
# print(get_data.recieve_data())