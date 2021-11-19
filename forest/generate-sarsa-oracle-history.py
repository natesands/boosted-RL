import gym
from gym import wrappers
import pickle
import numpy as np
from os import path



oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
oracle = 'cp-sarsa-agent-Q-04.pkl'
with open(path.join(oracle_dir, oracle), 'rb') as f:
    Q = pickle.load(f)

def max_action(Q, state):
    values = np.array([Q[state, a] for a in range(2)])
    action = np.argmax(values)
    return action


# discretize the space based on cartpole boundaries
pole_theta_space = np.linspace(-.20943951, 0.20943951, 12)
pole_theta_vel_space = np.linspace(-4, 4, 12)
cart_pos_space = np.linspace(-2.4, 2.4, 12)
cart_vel_space = np.linspace(-4, 4, 12)


def get_state(observation):
    cart_x, cart_x_dot, cart_theta, cart_theta_dot = observation
    cart_x = int(np.digitize(cart_x, cart_pos_space))
    cart_x_dot = int(np.digitize(cart_x_dot, cart_vel_space))
    cart_theta = int(np.digitize(cart_theta, pole_theta_space))
    cart_theta_dot = int(np.digitize(cart_theta_dot, pole_theta_vel_space))

    return (cart_x, cart_x_dot, cart_theta, cart_theta_dot)

def record_session(state_actions: list):
    fh = open(oracle_dir + '/history_10000.csv','w')
    for elem in state_actions:
        elem = [str(x) for x in elem]
        elem = ','.join(elem)
        fh.write(elem + '\n')
    fh.close()

env = gym.make("CartPole-v1")
# env.seed(0)
# np.random.seed(0)

outdir = oracle_dir + '/videos'
# env = wrappers.Monitor(env, outdir, force=True)
observation = env.reset()
state_action_pairs = []
t = 0
t_x = []
session = 1
for _ in range(10000):
    t += 1
    env.render()
    s = get_state(observation)

    action = max_action(Q,s)
    t_x.append(str(t) + ',' + str(observation[0]) + '\n')
    state_action_pairs.append(list(observation) + [action])
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        print(f'finished after {t} steps\n')
        # with open(problem_dir + '/' + 'session-' + str(session) + '.csv', 'w') as f:
        #     for pair in t_x:
        #         f.write(pair)
        session += 1
        t = 0
        t_x = []
env.close()
record_session(state_action_pairs)