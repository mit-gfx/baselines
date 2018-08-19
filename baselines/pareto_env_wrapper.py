from gym.envs.classic_control import CartPoleEnv
from gym.envs.mujoco import *
import numpy as np

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
        info['reward1'] = reward
        info['reward2'] = -np.dot(action, action)
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)


class InvertedDoublePendulumParetoWrapper(InvertedDoublePendulumEnv):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def step(self, action):
        # A direct copy of self.env.step(action)
        self.env.do_simulation(action, self.env.frame_skip)
        ob = self.env._get_obs()
        x, _, y = self.env.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.env.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        #r = alive_bonus - dist_penalty - vel_penalty
        r = alive_bonus
        done = bool(y <= 1)
        return ob, r, done, {}

    def reset(self):
        self.env.sim.reset()
        self.env.set_state(self.env.init_qpos, self.env.init_qvel)
        ob = self.env._get_obs()
        if self.env.viewer is not None:
            self.env.viewer_setup()
        return ob

    def render(self, mode='human'):
        self.env.render(mode)


class HopperParetoWrapper(HopperEnv):
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
        info['reward1'] = reward
        info['reward2'] = -np.dot(action, action)
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

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)

class Walker2dParetoWrapper(Walker2dEnv):
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
        info['reward1'] = reward
        info['reward2'] = -np.dot(action, action)
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)

class ReacherParetoWrapper(ReacherEnv):
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

class PusherParetoWrapper(PusherEnv):
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

class AntParetoWrapper(AntEnv):
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
        info['reward1'] = reward
        info['reward2'] = -np.dot(action, action)
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)
