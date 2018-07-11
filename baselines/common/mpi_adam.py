from mpi4py import MPI
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import IPython

class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None, loss=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float64')
        self.v = np.zeros(size, 'float64')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.loss = loss

    def update(self, localg, stepsize, ob, ac, atarg,vtarg, cur_lrmult):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float64')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()                                           
         #attempt line search
        cur_var_list = self.getflat()
        *initlosses, _ = self.loss(ob, ac, atarg,vtarg, cur_lrmult)
        if self.loss:
            while True:
                t = self.t + 1
                t = 1.0
                a = stepsize * np.sqrt(1 - self.beta2**t)/(1 - self.beta1**t)        
                m = self.beta1 * self.m + (1 - self.beta1) * globalg
                v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
                #step = (- a) * m / (np.sqrt(v) + self.epsilon)
                step = -stepsize * globalg
                self.setfromflat(cur_var_list + step)
                *newlosses, _ = self.loss(ob, ac, atarg,vtarg, cur_lrmult)
                if newlosses[2] > initlosses[2]:
                    stepsize /= 2
                else:
                    #print('break ' + str(stepsize))
                    #print('step ' + str(step))
                    break
        else:
            t = self.t + 1
            a = stepsize * np.sqrt(1 - self.beta2**t)/(1 - self.beta1**t)        
            m = self.beta1 * m + (1 - self.beta1) * globalg
            v = self.beta2 * v + (1 - self.beta2) * (globalg * globalg)
            step = (- a) * m / (np.sqrt(v) + self.epsilon)
        self.t = t
        self.m = m
        self.v = v
        self.setfromflat(cur_var_list + step)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

@U.in_session
def test_MpiAdam():
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float64'))
    b = tf.Variable(np.random.randn(2,5).astype('float64'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)
    do_update = U.function([], loss, updates=[update_op])

    tf.get_default_session().run(tf.global_variables_initializer())
    for i in range(10):
        print(i,do_update())

    tf.set_random_seed(0)
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a,b]
    lossandgrad = U.function([], [loss, U.flatgrad(loss, var_list)], updates=[update_op])
    adam = MpiAdam(var_list)

    for i in range(10):
        l,g = lossandgrad()
        adam.update(g, stepsize)
        print(i,l)
