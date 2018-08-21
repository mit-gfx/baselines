import IPython


#from cart_env_pytorch import CartScalarizationEnv
import pareto_env_wrapper as pw

import gym
from baselines import logger
from baselines.bench import Monitor
from gym.envs.mujoco import *

import argparse
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from matrix_io import *
import torch.autograd as autograd
from torch.autograd import Variable



parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--input_params', type=str, default=None, metavar='input_params')
parser.add_argument('--output_params', type=str, default='output_params.txt', metavar='output_params')
parser.add_argument('--sim', type=bool, default=False, metavar='sim')
parser.add_argument('--target1', type=float, default=0.0, metavar='target1')
parser.add_argument('--target2', type=float, default=0.0, metavar='target2')
parser.add_argument('--output_loss', type=str, default='output_loss.txt', metavar='output_loss')
parser.add_argument('--output_gradient', type=str, default='output_hessian.txt', metavar='output_gradient')
parser.add_argument('--output_hessian', type=str, default='output_gradient.txt', metavar='output_hessian')
parser.add_argument('--hessian', type=bool, default=False, metavar='hessian')
args = parser.parse_args()

input_params = args.input_params
output_params = args.output_params
sim = args.sim
targets = np.array([args.target1, args.target2])
output_loss = args.output_loss
output_gradient = args.output_gradient
output_hessian = args.output_hessian
need_hessian = args.hessian


'''
class Policy(nn.Module):
    def __init__(self, layers, obs_dim, act_dim):
        super(Policy, self).__init__()
        
        self.layers = []
        self.layers.append(nn.Linear(obs_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        #self.affine2 = nn.Linear(32, 32)
        #self.affine3 = nn.Linear(32, 32)
        self.layers.append(nn.Linear(32, act_dim * 2))

        self.saved_log_probs = []
        self.rollouts = []
        IPython.embed()


    def commit(self):
        self.rollouts.append(self.saved_log_probs)
        self.saved_log_probs = []

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        action_scores = x
        actions = action_scores.view(action_scores.numel())
        act_dim = env.get_action_dim()    
        return normal.Normal(actions[0:act_dim], torch.exp(actions[act_dim:]))
'''


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Policy, self).__init__()
        self.num_hidden = 2
        self.hpl = 8 #hidden per layer
        self.weights = []
        self.affine1 = nn.Linear(obs_dim, self.hpl)
        self.affine2 = nn.Linear(self.hpl, self.hpl)
        self.affine3 = nn.Linear(self.hpl, self.hpl)
        self.affine4 = nn.Linear(self.hpl, act_dim * 2)
        
        self.weights.append(self.affine1)
        self.weights.append(self.affine2)
        self.weights.append(self.affine3)
        self.weights.append(self.affine4)

        self.saved_log_probs = []
        self.rollouts = []


    def commit(self):
        self.rollouts.append(self.saved_log_probs)
        self.saved_log_probs = []

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        action_scores = torch.tanh(self.affine4(x))
        actions = action_scores.view(action_scores.numel())
        #act_dim = env.get_action_dim()    
        act_dim = env.action_space.shape[0]
        return dist.Normal(actions[0:act_dim], torch.exp(actions[act_dim:]))

#env = CartScalarizationEnv()
env = HopperEnv()
env = pw.HopperParetoWrapper(env)
#policy = Policy(env.get_obs_dim(), env.get_action_dim())
policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()




def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    m = policy(Variable(state))

    action = m.sample()

    policy.saved_log_probs.append(m.log_prob(action))
    #if torch.isnan(m.log_prob(action)):
    #    IPython.embed()
    return action.data.numpy().flatten()
    

def get_grads():
    
    grads = []
    for parameter in policy.parameters():
        grads.append(parameter.grad)
        parameter.grad = Variable(torch.zeros(parameter.grad.shape))
        
    return grads
    
    
def set_grads(grads1, weight_distance1, grads2, weight_distance2):
    global policy
    for grad1, grad2, parameter in zip(grads1, grads2, policy.parameters()):
        parameter.grad = grad1 * weight_distance1 + grad2 * weight_distance2
    

def flatten_hessian(loss_grad):
    model = policy
    cnt = 0
    
    #flat_grad = flatten_grad()
    #flat_params = flatten_params()
    
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):

        grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1        
        hessian[idx] = g2.data
    return hessian


def flatten_vector(vector):
    flat = []
    for component in vector:
        flat.append(component.view(component.numel()))
    return torch.cat(flat)

def flatten_grad():
    grads = []
    for parameter in policy.parameters():
        grads.append(parameter.grad.view(parameter.grad.numel()))
    return torch.cat(grads)


def flatten_params():
    params = []
    for parameter in policy.parameters():        
        params.append(parameter.data.numpy().flatten())
    flat_params = np.concatenate(params)
    flat_params = flat_params.reshape(flat_params.size, 1)
    return flat_params

def save_policy():
    flat_params = flatten_params()
    WriteMatrixToFile(output_params, flat_params)
    
def load_policy():       
    params = ReadMatrixFromFile(input_params)
    idx = 0
    all_params = list(policy.parameters())
    for i in range(len(all_params)):        
        parameter = all_params[i]
        numel = parameter.numel()        
        next_params = torch.reshape(torch.from_numpy(params[idx:idx+numel]), parameter.shape)
        #all_params[i] = next_params
        #TODO: can we just set parameterlist and make this easier?
        if i % 2 == 0:
            policy.weights[i // 2].weight = nn.Parameter(next_params)
        else:
            policy.weights[i // 2].bias = nn.Parameter(next_params)
        idx += numel
    

def update_params(observations, action_probs, returns1, returns2, sim):
    

    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions. For a Gaussian policy, it contains the mean and the logarithm of the standard deviation.
    
    #dist_info_vars = policy.dist_info_sym(observations_var)
    #design_dist_info_vars = design_policy.dist_info_sym()

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
    # distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
    # the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
    # rllab.distributions.DiagonalGaussian
    
    #dist = policy.distribution
    #design_dist = design_policy.distribution
    
    
    #action_lp = dist.log_likelihood_sym(actions_var, dist_info_vars)
    #design_lp = design_dist.log_likelihood_sym(designs_var, design_dist_info_vars)
    
    #lp =  action_lp + design_lp
    lp = action_probs
    
    
    returns1 = torch.from_numpy(returns1).type(torch.FloatTensor)
    returns2 = torch.from_numpy(returns2).type(torch.FloatTensor)
    
    #returns1 = (returns1 - returns1.mean()) / (returns1.std() + 1e-6)
    #returns2 = (returns2 - returns2.mean()) / (returns2.std() + 1e-6)
    
    
    #lp = torch.cat(lp)
    #IPython.embed()
    lp_sum = torch.stack([sum(sum(log_probs)) for log_probs in action_probs])
    
    #IPython.embed()
    #expectation1 = torch.mean(torch.exp(lp_sum) * returns1)
    #expectation2 = torch.mean(torch.exp(lp_sum) * returns2)
    
    
    
 
    #weight_distance1 = 2 * (expectation1 - targets[0])
    #weight_distance2 = 2 * (expectation2 - targets[1])
    #IPython.embed()
    
    weight_distance1 = 2 * (torch.mean(returns1) - targets[0])
    weight_distance2 = 2 * (torch.mean(returns2) - targets[1])
    
    

    # Note that we negate the objective, since most optimizers assume a minimization problem
    #surr1 = (TT.mean(lp * returns_var1) - targets[0])*(TT.mean(lp * returns_var1) - targets[0])
    #surr2 = (TT.mean(lp * returns_var2) - targets[1])*(TT.mean(lp * returns_var2) - targets[1])
    
    #surr = surr1 + surr2
    #IPython.embed()
    surr2 = torch.mean(lp_sum * Variable(returns2))        
    
    surr2.backward(create_graph=True, retain_graph=True)
    
    grads2 = get_grads()
    
    surr1 = torch.mean(lp_sum * Variable(returns1))
    
    surr1.backward(create_graph=True, retain_graph=True)
    
    grads1 = get_grads()
    
    
    

    set_grads(grads1, weight_distance1, grads2, weight_distance2)

    
    
    #And now we update
    loss = float((surr1 + surr2))
    grad = flatten_grad()
    print('grad norm is ', torch.norm(grad))
    hessian = None
    
    #policy_loss = surr1 + surr2
    if not sim:
        
        optimizer.step()
        optimizer.zero_grad()
    else:
        
        
        
        
        grad1_flat = flatten_vector(grads1)
        grad2_flat = flatten_vector(grads2)
                
        hessian1 = flatten_hessian(grads1)
        hessian2 = flatten_hessian(grads2)
        
        total_hess = weight_distance1 * hessian1 + 2.0 * torch.ger(grad1_flat, grad1_flat).data + \
                     weight_distance2 * hessian2 + 2.0 * torch.ger(grad2_flat, grad2_flat).data
        hessian = total_hess.numpy()

    grad = flatten_grad().data.numpy()
    policy.saved_log_probs = []
    policy.rollouts = []
    
    return loss, grad, hessian

def execute(sim=False):
    losses = []
    #load_policy()

       
    # We will collect 500 trajectories per iteration
    if not sim:
        N = 100
    else:
        N = 10
    # Each trajectory will have at most 100 time steps
    T = 100
    # Number of iterations
    n_itr = 10000
    # Set the discount factor for the problem
    discount = 1.0
    
    
    
    
    for iterate in range(n_itr):

        if iterate > 500:
            print('INCREASED SAMPLE SIZE')
            N = 100000
        print(iterate)
        paths = []
        i = 0
        for N_ in range(N):
            i += 1
            observations = []
            actions = []
            designs = []
            rewards1 = []
            rewards2 = []

            observation = env.reset()

            for _ in range(T):
                # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
                # sufficient statistics for the action distribution. It should at least contain entries that would be
                # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
                # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
                # not needed.
                action = np.array(select_action(observation))
                #design, _ = design_policy.get_action()

                #IPython.embed()
                #TODO: get the other policy here
                
                
                # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
                # case it is not needed.
                #TODO: two things:
                #First, get a noisy version of the next
                
                #inp = np.concatenate([action, np.array([1.0])]) #TODO not perfect but what can you do
                next_observation, reward, terminal, comp = env.step(action)
                
                
                
                if N_ == 0:
                    env.render()
                observations.append(observation)
                actions.append(action)
                #designs.append(design)        
                rewards1.append(comp[0])
                rewards2.append(comp[1])
                observation = next_observation
                if terminal:
                    # Finish rollout if terminal state reached
                    break
            
            policy.commit()
            # We need to compute the empirical return for each time step along the
            # trajectory
            returns1 = []
            return_so_far1 = 0
            for t in range(len(rewards1) - 1, -1, -1):
                return_so_far1 = rewards1[t] + discount * return_so_far1
            returns1.append(return_so_far1)
            # The returns are stored backwards in time, so we need to revert it
            #IPython.embed()
            returns1 = returns1[::-1]
            
            returns2 = []
            return_so_far2 = 0
            for t in range(len(rewards2) - 1, -1, -1):
                return_so_far2 = rewards2[t] + discount * return_so_far2
            returns2.append(return_so_far2)
            #IPython.embed()
            # The returns are stored backwards in time, so we need to revert it
            returns2 = returns2[::-1]
            
            

            paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                #designs=np.array(designs),
                rewards1=np.array(rewards1),
                returns1=np.array(returns1),
                rewards2=np.array(rewards2),
                returns2=np.array(returns2),
            ))
            
            if i > N:
                break
        #IPython.embed()
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        #designs = np.concatenate([p["designs"] for p in paths])
        returns1 = np.concatenate([p["returns1"] for p in paths])
        returns2 = np.concatenate([p["returns2"] for p in paths])
        #IPython.embed()
        loss, gradient, hessian = update_params(observations, policy.rollouts, returns1, returns2, sim) #TODO
        
        window_size = 10
        losses.append(loss)
        
        check = np.std(losses[-window_size:]) / np.mean(losses[-window_size:]) 
        
        
        
        #grad_norms = [np.linalg.norm(g_)**2 for g_ in g]
        #print('gradients are:', grad_norms)
        #print('total gradient is', np.sqrt(np.sum(grad_norms)))
        print('Average Return1:', np.mean([sum(p["rewards1"]) for p in paths]))
        print('Average Return2:', np.mean([sum(p["rewards2"]) for p in paths]))
        #print('Policy stats are:', policy.dist_info_sym(observations))
        
        print('check value is', check)
        
        if sim or (len(losses) >= 10 and check < 0.05) or iterate > 499:
            return loss, gradient, hessian
        
        
        
 
       


if __name__ == '__main__':
    if input_params != None:
        load_policy()
    loss, gradient, hessian = execute(sim=True)
    save_policy()

    WriteMatrixToFile(output_loss, np.array([loss]))
    WriteMatrixToFile(output_gradient, gradient)
    if hessian is not None and need_hessian:
        WriteMatrixToFile(output_hessian, hessian)
    
    
    

    
