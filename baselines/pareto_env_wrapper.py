from gym.envs.classic_control import CartPoleEnv
from gym.envs.mujoco import SwimmerEnv

class CartPoleParetoWrapper(CartPoleEnv):
    env = None
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def seed(self, seed=None):
        ret = self.env.seed(seed)
        self.np_random = self.env.np_random
        return ret

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

class SwimmerParetoWrapper(SwimmerEnv):
    env = None
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def seed(self, seed=None):
        ret = self.env.seed(seed)
        self.np_random = self.env.np_random
        return ret

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)