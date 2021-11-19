import numpy as np
import gym
import pickle
import _policies
from os import path
from gym import wrappers
from stochastic_arbor import *

work_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/forest/workdir'


# record state action pairs ----------------
def record_session(state_actions: list):
    fh = open(tmp_dir + '/session_data', 'w')
    for elem in state_actions:
        elem = [str(x) for x in elem]
        elem = ','.join(elem)
        fh.write(elem + '\n')
    fh.close()


# record video or no -----------------------
def record_video(i: int) -> bool:  # returns True if episode i should be recorded
    # return i < 5
    return False

outdir = path.join(work_dir, 'videos')

# train trees -------------------------------
oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
train_data = []
with open(path.join(oracle_dir, 'train_data.csv'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        line = [float(s) for s in line]
        train_data.append(line)
train_data = np.array(train_data)

T_oracle = RLTree(train_data, max_depth=2)
tree_rules = T_oracle.get_rules()
print(f'Initialized Tree (max depth = {T_oracle.max_depth}) with rules:\n')
for rule in tree_rules:
    print(rule)

# initialize gym environment -------------------
env = gym.make("CartPole-v2")
env = wrappers.Monitor(env, outdir, video_callable=record_video, force=True)
observation = env.reset()
state_action_pairs = []
t = 0
for _ in range(1000):
    t += 1
    env.render()
    action = T_oracle.act(observation.reshape(1,-1))
    #action = F.vote(observation.reshape(1, -1))
    # state_action_pairs.append(list(observation) + [action])
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        print(f'{t} steps\n')
        t = 0
# record_session(state_action_pairs)
env.close()
