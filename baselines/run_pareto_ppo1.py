#!/usr/bin/env python3

from baselines.common.cmd_util import mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import pareto_env_wrapper as pw

import os
from mpi4py import MPI
import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
import IPython

from gym.envs.mujoco import *


def make_pareto_mujoco_env(env_id, seed, target1, target2, target3):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    #TODO: generalize these targets
    
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    if env_id == 'InvertedDoublePendulum-v2':
        env = InvertedDoublePendulumEnv()
        env = pw.InvertedDoublePendulumParetoWrapper(env)
    elif env_id == 'InvertedPendulum-v2':
        env = InvertedPendulumEnv()
        env = pw.InvertedPendulumParetoWrapper(env, target1, target2)
    elif env_id == 'Hopper-v2':
        env = HopperEnv()
        env = pw.HopperParetoWrapper(env)
    elif env_id == 'Swimmer-v2':
        # TODO: fix nan.
        env = SwimmerEnv()
        env = pw.SwimmerParetoWrapper(env)
    elif env_id == 'Walker2d-v2':
        env = Walker2dEnv()
        env = pw.Walker2dParetoWrapper(env)
    elif env_id == 'Reacher-v2':
        # TODO: fix nan.
        env = ReacherEnv()
        env = pw.ReacherParetoWrapper(env)
    elif env_id == 'Pusher-v2':
        # TODO: fix nan.
        env = PusherEnv()
        env = pw.PusherParetoWrapper(env)
    elif env_id == 'Ant-v2':
        env = AntEnv()
        env = pw.AntParetoWrapper(env)
    else:
        raise ValueError('%s is not supported yet.' % env_id)
    env = Monitor(env, os.path.join(logger.get_dir(), str(rank)))
    env.seed(seed)    
    env.target1 = target1
    env.target2 = target2
    env.target3 = target3
    return env


def train(env_id, num_timesteps, seed, target1, target2, target3, output_prefix, input_file, model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple, pareto_mlp_policy
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
    #    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #        hid_size=64, num_hid_layers=2)
        # TODO: add the filepath.
        return pareto_mlp_policy.ParetoMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=10, num_hid_layers=1, file_path='common/test.txt')
    env = make_pareto_mujoco_env(env_id, seed, target1, target2, target3)
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            input_file=input_file,
            output_prefix=output_prefix,
            timesteps_per_actorbatch=16384,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=1e-2, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear'
        )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi

def main():
    logger.configure()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path')
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError('You have to provide a model path.')
    
    if not args.play:
        # train the model
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path, target1 = args.target1, target2 = args.target2, target3 = args.target3, output_prefix = args.output_prefix, input_file = args.input_file)
    else:       
        # construct the model object, load pre-trained model and render
        pi = train(args.env, num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)
        env = make_pareto_mujoco_env(args.env, seed=0)

        ob = env.reset()
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ =  env.step(action)
            env.render()
            if done:
                ob = env.reset()

if __name__ == '__main__':
    main()
