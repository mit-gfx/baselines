import numpy as np



goal_x = 0.5
R1 = True
class CartScalarizationEnv():

    def get_obs_dim(self):
        return len(self.get_current_obs())
        
    def get_action_dim(self):
        return 1 #TODO: generalize

    def __init__(self, *args, **kwargs):
        self.H = 100
        self.reset_range = 0.05
        self.design_dim = 1
        self.position = 0.0
        self.velocity = 0.0

       

    def get_current_obs(self):
        return np.array([self.position, self.velocity])

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.idx = 0
        return self.get_current_obs()


    def render(self):
        pass
        #print('position is', self.position)
        #print('velocity is', self.velocity)
        #print('action was', self.action)


    def step(self, inp):

        design = inp[-self.design_dim:]
        action = inp[:self.design_dim]
        #if action[0] < -0.4:
        #    action[0] = -0.4
        #if action[0] > 0.4:
        #    action[0] = 0.4

        self.design = design
        
        
        reward_computer = self.compute_reward(action)

        self.internal_step(action, design)
            
            
        # notifies that we have stepped the world
        next(reward_computer)
        # actually get the reward
        reward = next(reward_computer)
        done = self.is_current_done()
        next_obs = self.get_current_obs()

        return next_obs, reward, done, self.get_rewards(action)
        
    '''
    def get_next_state(self, action):
        #Before stepping, we should return the new state:
        dt = 0.5
        vel = self.lineaVelocity[0] + dt * action[0] / self.design[0]
        pos = self.cart.position[0] + dt * self.cart.linearVelocity[0]
        
        return TT.stack(vel, pos)
    '''
    
    def internal_step(self, action, design):       
        dt = 0.5
        #IPython.embed()
        self.velocity += dt * action[0] / self.design[0]
        self.position += dt * self.velocity
        self.idx += 1

    def get_rewards(self, action):
        #TODO: is design even used here?
        done = np.abs(self.position - goal_x) < 0.1 and np.abs(self.velocity) < 0.1
        expire = self.idx == self.H - 1
        if done:
            reward1 = 0.0
            reward2 = 0.0
        elif expire:
            reward1 = -np.abs(self.position - goal_x) - np.abs(self.velocity) - 1.0
            reward2 = -np.abs(self.position - goal_x) - np.abs(self.velocity) - 1.0
        else:
            reward1 = -1.0
            reward2 = -action * action
        self.action = action
        return np.array([reward1, reward2])
        


    def compute_reward(self, action):
        yield        
        rewards = self.get_rewards(action)
        yield np.linalg.norm(rewards)**2


    def is_current_done(self):
        done = np.abs(self.position - goal_x) < 0.1 and np.abs(self.velocity) < 0.1
        expire = self.idx == self.H - 1
        if done:
            pass
            #print('goal')
        return done or expire

