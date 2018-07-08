from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import IPython
from baselines.common.math_util import ReadMatrixFromFile, WriteMatrixToFile
from functools import reduce
from operator import mul
import os
import random

np.set_printoptions(threshold=np.nan)

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float64')
    vpreds = np.zeros(horizon, 'float64')
    news = np.zeros(horizon, 'int64')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        ac = ac * 0.0 + 1.0
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float64')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def get_model_vars(pi):
    variables = pi.get_trainable_variables()
    ret_vars = []
    for variable in variables:
        ret_vars.extend(list(np.ndarray.flatten(variable.eval())))
    return ret_vars
    

def get_num_dims(shape):
    list_shape = list(shape)
    cast_list_shape = [int(x) for x in list_shape]
    return reduce(mul, cast_list_shape, 1)

def get_gradient_indices(pi):
    variables = pi.get_trainable_variables()
    #TODO: generalize this to other networks
    idx = 0
    ret_vars = []
    for var in variables:
        num_vars = get_num_dims(var.shape)
        if 'pi/pol' in var.name:            
            ret_vars.extend(range(idx, idx + num_vars))   
        
        idx += num_vars #TODO: Move this to utils
    return ret_vars
        

def return_routine(pi, d, batch, output_prefix, losses, cur_lrmult, lossandgradandhessian, gradients, hessians, gradient_set):
    gradient_indices = get_gradient_indices(pi)
    
    if hessians:
        hessian_set = []
        
        for batch in d.iterate_once(d.n):
            *newlosses, g, h = lossandgradandhessian(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            hessian_set.extend(h)
        mean_hessian = np.mean(hessian_set, axis=0)
        mean_hessian = mean_hessian[gradient_indices, gradient_indices]
        WriteMatrixToFile(output_prefix + '_hessian.bin', mean_hessian)
    if gradients:
        mean_gradient = np.mean(gradient_set, axis=0)
        mean_gradient = mean_gradient[gradient_indices]
        WriteMatrixToFile(output_prefix + '_gradient.bin', mean_gradient)
    mean_objective = np.sum(np.mean(losses, axis=0)[0:3])
    WriteMatrixToFile(output_prefix + '_objective.bin', np.array([[mean_objective]]))
    
    model_vars = np.array(get_model_vars(pi))[gradient_indices]
    WriteMatrixToFile(output_prefix + '_vars.bin', np.array(model_vars))
    
    

    

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        gradients=True,
        hessians=False,
        model_path='model',
        output_prefix,
        sim):
        
    #Directory setup:
    first_iter = True
    model_dir = 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float64, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float64, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float64, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    #total_loss = pol_surr + pol_entpen + vf_loss
    total_loss = pol_surr * 0.0 + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    

    
    lossandgradandhessian = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list), U.flathess(total_loss, var_list)])
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon, loss=lossandgrad)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    #optimizer = ScipyOptimizerInterface(lossandgrad, method='L-BFGS-B')

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    # Set the logs writer to the folder /tmp/tensorflow_logs
    tf.summary.FileWriter('/home/aespielberg/ResearchCode/baselines/baselines/tmp/',
        graph_def=tf.get_default_session().graph_def)
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=False)    
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"


    gradient_indices = get_gradient_indices(pi)
    
    while True:
        if callback: callback(locals(), globals())
        
        #ANDYTODO: add new break condition
        '''
        try:
            print(np.std(rewbuffer) / np.mean(rewbuffer))
            print(rewbuffer)
            if np.std(rewbuffer) / np.mean(rewbuffer) < 0.01: #TODO: input argument
                break
        except:
            pass #No big
        '''
        
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        #IPython.embed()

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        #print(seg['ob'])
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data                
        #for _ in range(optim_epochs):
        while True:
            gradient_set = []
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            '''
            if first_iter:
                holdout_batch = list(d.iterate_once(d.n))[0].copy()
                holdout_cur_lrmult = cur_lrmult
                first_iter = False
            else:
                elems = list(d.iterate_once(d.n))[0].copy()
                holdout_batch["ob"] = np.append(holdout_batch["ob"], elems["ob"], axis=0)
                holdout_batch["ac"] = np.append(holdout_batch["ac"], elems["ac"], axis=0)
                holdout_batch["atarg"] = np.append(holdout_batch["atarg"], elems["atarg"], axis=0)
                holdout_batch["vtarg"] = np.append(holdout_batch["vtarg"], elems["vtarg"], axis=0)
            '''
            for batch in d.iterate_once(optim_batchsize):                
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                gradient_set.append(g)
                if not sim:
                    #train = optimizer.minimize(log_x_squared)                     
                    adam.update(g, optim_stepsize * cur_lrmult, batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    print(str(np.linalg.norm(g)) + ' ' + str(newlosses[2]))
                losses.append(newlosses)
            #logger.log(fmt_row(13, np.mean(losses, axis=0)))
            if np.linalg.norm(g) < 3e-3:
                logger.log(fmt_row(13, np.mean(losses, axis=0)))                
                break
        #*holdout_newlosses, holdout_g = lossandgrad(holdout_batch["ob"], holdout_batch["ac"], holdout_batch["atarg"], holdout_batch["vtarg"], holdout_cur_lrmult)
        print('objective is')
        print(np.sum(newlosses[0:3]))  
        print('gradient is')
        print(np.mean(list(map(np.linalg.norm, np.array(g))))) 
        print('data size is ')
        #print(holdout_batch['ac'].shape)
        #print(get_model_vars(pi))
        if sim:
            print('return routine')
            return_routine(pi, d, batch, output_prefix, losses, cur_lrmult, lossandgradandhessian, gradients, hessians, gradient_set)            
            return pi
        if np.mean(list(map(np.linalg.norm, gradient_set))) < 1e-4: #TODO: make this a variable
            #TODO: abstract all this away somehow (scope)
            print('minimized!')
            return_routine(pi, d, batch, output_prefix, losses, cur_lrmult, lossandgradandhessian, gradients, hessians, gradient_set)
            return pi        
        logger.log("Evaluating losses...")
        losses = []        
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        if iters_so_far > 1:
            U.save_state(model_dir + model_path + str(iters_so_far))

    print('out of time')
    return_routine(pi, d, batch, output_prefix, losses, cur_lrmult, lossandgradandhessian, gradients, hessians, gradient_set)
    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
