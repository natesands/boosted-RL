import gym
import numpy as np
from stochastic_arbor import *
import pickle


# run sarsa agent from starting state, return history and accumulated reward
def sarsa_agent_rollout(Q: dict, env, start_state=None, random_seed=None, render=False) -> tuple:


    # discretize the space based on cartpole boundaries
    pole_theta_space = np.linspace(-.20943951, 0.20943951, 12)
    pole_theta_vel_space = np.linspace(-4, 4, 12)
    cart_pos_space = np.linspace(-2.4, 2.4, 12)
    cart_vel_space = np.linspace(-4, 4, 12)

    def max_action(Q, state):
        values = np.array([Q[state, a] for a in range(2)])
        action = np.argmax(values)
        return action

    def get_state(observation):
        cart_x, cart_x_dot, cart_theta, cart_theta_dot = observation
        cart_x = int(np.digitize(cart_x, cart_pos_space))
        cart_x_dot = int(np.digitize(cart_x_dot, cart_vel_space))
        cart_theta = int(np.digitize(cart_theta, pole_theta_space))
        cart_theta_dot = int(np.digitize(cart_theta_dot, pole_theta_vel_space))

        return (cart_x, cart_x_dot, cart_theta, cart_theta_dot)

    if random_seed is not None:
        env.seed(random_seed)
        np.random.seed(0)

    if start_state is None:
        observation = env.reset()
    else:
        observation = env.reset_to_state(start_state)
    if render:
        env.render()

    state_action_history = []

    s = get_state(observation)
    action = max_action(Q, s)
    observation, reward, done, info = env.step(action)
    state_action_history.append(list(observation)+[action])
    total_reward = reward
    while not done:
        if render:
            env.render()
        s = get_state(observation)
        action = max_action(Q, s)
        observation, reward, done, info = env.step(action)
        state_action_history.append(list(observation)+[action])
        total_reward += reward
    env.close()
    return np.array(state_action_history), total_reward

# ------------------------StochasticArbor---------------------------

def load_history(csv_file):
    history = np.genfromtxt(csv_file, delimiter=',')
    return history

def run_tree_agent(T: RLTree, env, start_state=None,  render=False):
    if start_state is None:
        start_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    observation = env.reset_to_state(start_state)
    if render:
        env.render()
    action = T.act(observation.reshape(1, -1))
    observation, reward, done, info = env.step(action)
    total_reward = reward
    while not done:
        if render:
            env.render()
        action = T.act(observation.reshape(1, -1))
        observation, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward

def run_forest_agent(F: StochasticArbor, env, start_state=None, render=False):
    if start_state is None:
        start_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    observation = env.reset_to_state(start_state)
    if render:
        env.render()
    action = F.vote(observation.reshape(1, -1))
    observation, reward, done, info = env.step(action)
    total_reward = reward
    while not done:
        if render:
            env.render()
        action = F.vote(observation.reshape(1, -1))
        observation, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward

# --------------------------------CEM--------------------------------------------

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

def run_cem_agent(cem_agent: BinaryActionLinearPolicy, env, start_state=None, render=False):
    if start_state is None:
        start_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    observation = env.reset_to_state(start_state)
    if render:
        env.render()
    action = cem_agent.act(observation)
    observation, reward, done, info = env.step(action)
    total_reward = reward
    while not done:
        if render:
            env.render()
        action = cem_agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward

if __name__ == '__main__':

    workdir = '/Users/ironchefnate/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/experiments/boosted_forest_v2'
    forest_file = 'Forest_depth4_samples300_cp2_00.pkl'
    # workdir = '/Users/ironchefnate/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/CSCI_699_HRI/project/code/robolog/GA/cem/augmented-reward/zero-distance'
    # agent_file = 'agent-0019.pkl'
    # history_file = 'zero-distance-agent-history-10k.csv'
    with open(path.join(workdir, forest_file), 'rb') as f:
        F = pickle.load(f)


    env = gym.make("CartPole-v2")
    # max_depth = 11
    # zero_distance_agent_history = load_history(path.join(workdir, history_file))
    # zero_d_tree = RLTree(zero_distance_agent_history,max_depth=max_depth)
    # # run_tree_agent(zero_d_tree, env, render=True)
    # rewards = []
    # for i in range(100):
    #     #if i % 10 == 0:
    #     #     render = True
    #     # else:
    #     #     render = False
    #     reward = run_tree_agent(zero_d_tree, env, render=False)
    #     print(reward)
    #     rewards.append(reward)
    # print(f'depth {max_depth}')
    # print(f'num rules: {len(zero_d_tree.get_rules())}')
    # print(f'mean reward: {np.mean(rewards)},  max/min: {np.max(rewards)}, {np.min(rewards)}, var: {np.var(rewards)}'
    #       f', std: {np.std(rewards)}')





    # rewards = []
    # for i in range(100):
    #     #if i % 10 == 0:
    #     #     render = True
    #     # else:
    #     #     render = False
    #     reward = run_cem_agent(path.join(workdir, agent_file), env, render=False)
    #     print(reward)
    #     rewards.append(reward)
    # print(f'mean reward: {np.mean(rewards)},  max/min: {np.max(rewards)}, {np.min(rewards)}, var: {np.var(rewards)}'
    #       f'std: {np.std(rewards)}')


    # F_rewards = np.array([])
    # for j in range(100):
    #     if j % 20 == 0 :
    #         render = True
    #     else:
    #         render = False
    #     reward = run_forest_agent(F, env, render=render)
    #     F_rewards = np.append(F_rewards, reward)
    # print(f"Avg reward: {np.mean(F_rewards)}")
    #
    #
    forest_mean_rewards = np.array([])
    for i in range(1,len(F.trees)+1):
    #for i in range(1,3):
        subF = StochasticArbor()
        subF.trees = F.trees[0:i]
        subF_rewards = np.array([])

        for j in range(100):
            # if j == 0: render=True
            # else: render=False
            reward = run_forest_agent(subF, env, render=False)
            subF_rewards = np.append(subF_rewards, reward)

        forest_mean_rewards = np.append(forest_mean_rewards, np.mean(subF_rewards))
        print(f'Forest with {i} trees, avg reward: {np.mean(subF_rewards)}, max/min: '
              f'{np.max(subF_rewards)},{np.min(subF_rewards)}, var: {np.var(subF_rewards)}, '
              f'std: {np.std(subF_rewards)}')
    print(f'mean rewards: {forest_mean_rewards}')
    #
    #
    #
    # #
    # for i in range(len(F.trees)):
    #     tree_rewards = np.array([])
    #     for j in range(100):
    #         # if j == 0: render=True
    #         # else: render=False
    #         reward = run_tree_agent(F.trees[i], env, render=False)
    #         tree_rewards = np.append(tree_rewards, reward)
    #
    #     print(f'Tree {i}, beta: {F.trees[i].beta}, avg reward: {np.mean(tree_rewards)}')

    #
    #

    # for i in range(5):
    #     run_forest_agent(F, env, render=True)
    # problem_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/synth/cp-sarsa-agent/cp-sarsa-agent-03/agent-04'
    #
    # with open(problem_dir + '/cp-sarsa-agent-Q-04.pkl', 'rb') as f:
    #     Q = pickle.load(f)
    #
    # env = gym.make("CartPole-v2")
    # rewards = []
    # render = False
    # for i in range(100):
    #     if i % 10 == 0:
    #         render = True
    #     else:
    #         render = False
    #     _, reward = sarsa_agent_rollout(Q, env, render=render)
    #     print(reward)
    #     rewards.append(reward)
    # print(f'mean reward: {np.mean(rewards)},  max/min: {np.max(rewards)}, {np.min(rewards)}, var: {np.var(rewards)}'
    #       f'std: {np.std(rewards)}')
    env.close()
