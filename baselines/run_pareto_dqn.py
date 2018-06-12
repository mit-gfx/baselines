import os
import sys
import argparse

import gym
from baselines import deepq

import pareto_env_wrapper as pw

def dqn_callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def run_dqn(output_dir, train):
    # Create the environment.
    env = gym.make('CartPole-v0')
    env = pw.CartPoleParetoWrapper(env)
    saved_file = os.path.join(output_dir, 'cartpole_dqn.pkl')
    if train:
        model = deepq.models.mlp([64])
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,                        # Recommended: 1e-3
            max_timesteps=100000,           # Recommended: 100000
            buffer_size=50000,              # Recommended: 50000
            exploration_fraction=0.1,       # Recommended: 0.1
            exploration_final_eps=0.02,     # Recommended: 0.02
            print_freq=10,                  # Recommended: 10
            callback=dqn_callback
        )
        # Save file.
        print('[dqn] Saving model to', saved_file)
        act.save(saved_file)
    else:
        act = deepq.load(saved_file)

    # Replay.
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
    print("[dqn] Episode reward", episode_rew)


# Usage: python run_pareto --baseline <baseline_name> --output <output_dir> --train True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', help='baseline names', type=str, default='dqn')
    parser.add_argument('--output', help='output folder', type=str, default='pareto_dqn')
    parser.add_argument('--train', help='Train or replay', type=str, default='True')
    args = parser.parse_args()
    
    # Check inputs.
    all_baselines = ['dqn']
    if args.baseline not in all_baselines:
        raise ValueError('Baseline %s is not supported yet.' % args.baseline)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.train not in ['True', 'False']:
        raise ValueError('Train flag %s is not supported. Should be either True or False' % args.train)
    train_flag = args.train == 'True'

    if args.baseline == 'dqn':
        run_dqn(args.output, train_flag)