import time, utils
import gym

from brain import Brain
from environment import Environment
from optimizer import Optimizer

# Global variables
ENV = 'CartPole-v0'
OUTPUT_NAME = 'models/A3C_'+ENV 
MAX_EPISODE_STEPS = 10000                 # Number of steps to consider the game is won (default = 200)
gym.envs.registry.env_specs['CartPole-v0'].max_episode_steps = MAX_EPISODE_STEPS

TRAINING_TIME = 10
N_THREADS = 8
N_OPTIMIZERS = 4


EPS_START = 0.
EPS_END   = 0.
EPS_STEPS = 100000

GAMMA = 0.99

def main():

    state_shape, n_actions = utils.getSpecs(ENV)
    brain = Brain(state_shape, n_actions, GAMMA)
    #brain = Brain(state_shape, n_actions, GAMMA)

    envs = [Environment(brain, ENV, EPS_START, EPS_END, EPS_STEPS) for i in range(N_THREADS)]
    opts = [Optimizer(brain)   for i in range(N_OPTIMIZERS)]

    for opt in opts:
        opt.start()
    for env in envs:
        env.start()

    time.sleep(TRAINING_TIME)

    for opt in opts:
        opt.stop()
    for opt in opts:
        opt.join()
    for env in envs:
        env.stop()
    for env in envs:
        env.join()

    print("--------------")
    print("Training done!")
    print("Saving model under:",OUTPUT_NAME)
    brain.save(OUTPUT_NAME)
    print("--------------")
    print("Testing")
    test_env    = Environment(brain, ENV, render=True)
    test_env.run()

if __name__ == "__main__":
    main()
