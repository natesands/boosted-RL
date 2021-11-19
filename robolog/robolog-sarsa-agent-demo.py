from cartpole_agent import *
import gym
from gym import wrappers

problem_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/synth/cp-sarsa-agent'

A = CartPoleAgent(problem_dir, 'cp-sarsa-data-01', 10)
for rule in A.rules:
    print(rule)

env = gym.make("CartPole-v1")
# env.seed(0)
# np.random.seed(0)

outdir = problem_dir + '/robolog-sarsa-agent/results'
env = wrappers.Monitor(env, outdir, force=True)
observation = env.reset()
state_action_pairs = []
for _ in range(2000):
    env.render()
    action = A.act(observation)
    state_action_pairs.append(list(observation) + [action])
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
clean_problem_dir(problem_dir)

