import numpy as np
import random 

N_STEP_RETURN = 8

FRAME = 0

class Agent:

    def __init__(self, brain, eps_start, eps_end, eps_steps):
        self.brain = brain
        self.eps_start = eps_start
        self.eps_end   = eps_end   
        self.eps_steps = eps_steps
        self.memory = []
        self.R = 0.

    def epsilon(self):
        if FRAME > self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start - (self.eps_start - self.eps_end) / self.eps_steps * FRAME

    def act(self,s,veto_a=None):
        global FRAME; FRAME = FRAME+1
        if random.random() < self.epsilon():
            return random.randint(0, self.brain.n_actions-1)
        else:
            s = np.asarray([s])
            p = self.brain.predict_p(s)
            p = p[0]
            if veto_a!=None:
                p_veto_a = p[veto_a]
                p[veto_a] = 0
                for a in range(self.brain.n_actions):
                    if a == veto_a: continue
                    p[a]+= p_veto_a / (self.brain.n_actions-1) # divide p_veto_a among other actions
            return np.random.choice(range(self.brain.n_actions), p=p)


    def train(self,s,a,r,s_,done,total_r):
        def get_sample(memory, n):
            s, a, _, _, _, _ = memory[0]
            _, _, _, s_, _, _ = memory[n-1]

            return s, a, self.R, s_, done, total_r

        a_cats = np.zeros(self.brain.n_actions)  # turn action into one-hot representation
        a_cats[a] = 1 

        self.memory.append( (s, a_cats, r, s_, done, total_r) )

        self.R = ( self.R + r * self.brain.gamma**N_STEP_RETURN ) / self.brain.gamma

        if done is True:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_, done, total_r = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_, done, total_r)

                self.R = ( self.R - self.memory[0][2] ) / self.brain.gamma
                self.memory.pop(0)      

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_, done, total_r = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_, done, total_r)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)  
