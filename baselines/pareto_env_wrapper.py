from gym.envs.classic_control import CartPoleEnv
from gym.envs.mujoco import SwimmerEnv
from gym.envs.mujoco import InvertedPendulumEnv
from gym.envs.mujoco import InvertedDoublePendulumEnv

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


class InvertedPendulumParetoWrapper(InvertedPendulumEnv):
    env = None
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset_model(self):
        return self.env.reset_model()

    def viewer_setup(self):
        return self.env.viewer_setup()


class InvertedDoublePendulumParetoWrapper(InvertedDoublePendulumEnv):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)

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