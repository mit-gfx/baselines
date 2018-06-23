from gym.envs.classic_control import CartPoleEnv
from gym.envs.mujoco import *
import IPython
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


    def __init__(self, env, target1, target2):
        self.target1 = target1
        self.target2 = target2
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata
        self.total_energy = 0.0

    def step(self, action):
        # TAO: @Andy you can clamp action here.
        ob, reward, done, info = self.env.step(action)
        
        #General rule:
        #Want to optimize: 0.5 [(||y_target - y)^2 + (u_target  - u)^2]
        #In order to do this, make the reward the time differential:
        #(y_target - y) * ydot + (u_target - u) * udot
        target_act = self.target2
        target_y = self.target1       
        
        #TODO: get targets
        
        dt = self.env.dt
        y = np.cos(ob[1])
        ydot = -np.sin(ob[1]) * ob[3]
        u = action[0]
        
        act = 0.5 * u * u * dt
        
        self.total_energy += act

        actdot = 0.5 * u * u
        
        reward1 = (y - target_y)
        reward2 = (self.total_energy - target_act)
        
        total_reward = -reward1 * ydot + -reward2 * actdot
        
        
        
        # TAO: @Andy you can recalculate reward here.
        # You can unpack ob to get positions and velocities. Read
        # gym/env/mujoco/inverted_pendulum.py to see the full definition of ob.
        return ob, total_reward, done, info

    def reset(self):
        self.total_energy = 0.0
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
        ob, reward, done, info = self.env.step(action)
        # TODO: Recalculate reward.
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

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
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)
