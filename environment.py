import gym, threading, time, utils
from agent import Agent

MAX_REPEAT_ACTION = 10
THREAD_DELAY = 0.001 # Yield time

class Environment(threading.Thread):

    def __init__(self, brain, environment, eps_start=0, eps_end=0, eps_steps=0, render=False):
        threading.Thread.__init__(self)
        self.env         = gym.make(environment)
        self.stop_signal = False
        self.render      = render
        self.agent       = Agent(brain,eps_start,eps_end,eps_steps)


    def runGame(self):
        R = 0
        s = utils.process(self.env.reset(),self.env.spec.id)

        n_a   = 0
        old_a = None

        while True:
            time.sleep(THREAD_DELAY)

            if self.render: self.env.render()

            if n_a > MAX_REPEAT_ACTION:
                a = self.agent.act(s,old_a)
            else:
                a = self.agent.act(s)

            if a == old_a:
                n_a+= 1
            else:
                n_a = 0

            old_a = a

            s_, r, done, info = self.env.step(a)
            s_ = utils.process(s_,self.env.spec.id)
            R+= r
            self.agent.train(s,a,r,s_,done,R)
            
            s = s_

            if done or self.stop_signal:
                break
        print("Score:",R)

    def run(self):
        while not self.stop_signal:
            self.runGame()

    def stop(self):
        self.stop_signal = True