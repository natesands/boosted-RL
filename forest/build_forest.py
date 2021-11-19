"""
get a distribution for oracle's statessample N number of states S = s_1, ..., s_N
do rollouts of oracle on N states and compute total reward --
take the K states with the largest reward, and train K trees starting from that state
initialize weights w_i = 1,  i = 1, ..., N
find the tree t_1 that minimizes sum(w_i I[O(s_i) - t_1(s_i) > 0])
add t_1 to forest
calculate error
calculate alpha
calculate beta
update weights
repeat

Parameters
----------
N           number of starting states to sample from oracle's distribution over states
n_iter      number of rollouts used to calculate expected reward given a starting state
pool_size   number of trees in candidate pool
max_depth   max depth of trees
target      version of CartPole (v0: max 200 iters, v1: max 500 iters, v2:  max 10,000 iters???)
sample_size number of oracle states to sample from
"""
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from sklearn.neighbors import KernelDensity
from os import path
from agent_rollouts import *
import pickle
from train_cem_agent import *
from stochastic_arbor import *


# generates 3D contour graph of kde distribution for two features
def kde_contour(f1, f2):
    # 3D contour of distribution over 2 features
    data = np.column_stack((f1, f2))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    x_max = np.max(f1)
    x_min = np.min(f1)
    y_max = np.max(f2)
    y_min = np.min(f2)
    x, y = np.meshgrid(np.arange(x_min - 1, x_max + 1, .1), np.arange(y_min - 1, y_max + 1, .1))
    points = np.column_stack([x.ravel(), y.ravel()])
    prob = np.exp(kde.score_samples(points))
    prob = prob.reshape(x.shape)
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, prob)
    plt.show()


params = dict(N=3, n_iter=10, pool_size=10, max_depth=2, target="CartPole-v1", forest_size=5, sample_size=10000)

work_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/forest/workdir'
oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
oracle_file = 'cp-sarsa-agent-Q-04.pkl'
oracle_history_file = 'history_10000_cpv1.csv'
with open(path.join(oracle_dir, oracle_file), 'rb') as f:
    Q = pickle.load(f)

# initiate gym environment ----------------------------------
env = gym.make(params['target'])

#  calculate distribution of oracle states and generate N sample points -----------------------------------
oracle_history = np.genfromtxt(path.join(oracle_dir, oracle_history_file), delimiter=',')
oracle_history = oracle_history[:, :-1]
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(oracle_history)
sample_states = kde.sample(params['N'])
# INCLUDE gym env normal starting state:
normal_start_state = env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
sample_states = np.vstack((sample_states, normal_start_state))
print("states sampled from oracle distribution:")
print(sample_states)
# for each sample point, run oracle, record history and reward
oracle_rollouts = []
oracle_rollout_rewards = np.array([])
print("Performing rollouts of oracle on each state")
for state in sample_states:
    print(state)
    history, reward = sarsa_agent_rollout(Q, env, state, render=False)
    oracle_rollouts.append(history)
    oracle_rollout_rewards = np.append(oracle_rollout_rewards, reward)
    print(f'Reward: {reward}')
# generate n_iter cem agents for each starting state
print("generating cem agents for each state")
cem_agents = []  # will contain N lists, each containing cem agents trained for a particular start state
tree_pool = []
for state in sample_states:
    print(f"training cem agent for state {state}")
    thetas, means = train_agents(env, start_state=state, n_iter=20, n_steps=500, render=False)
    for theta in thetas:
        cem_agents.append([BinaryActionLinearPolicy(theta) for theta in thetas])
        agent = BinaryActionLinearPolicy(theta)
        print(f"generating history of cem agent for state {state}")
        history = generate_history(env, agent, start_state=state, num_iter=4000, render=False)
        print(f'creating new tree...')
        new_tree = RLTree(history, params['max_depth'])
        print(f'running tree model')
        run_tree_agent(new_tree, env, start_state=state, render=True)
        tree_pool.append(new_tree)
    # agent = BinaryActionLinearPolicy(thetas[-1])
    # print(f"generating history of best agent for state {state}")
    # history = generate_history(env, agent, start_state=state, num_iter=4000, render=False)
    # print(f'Training tree agent')
    # new_tree = RLTree(history, params['max_depth'])
    # print(f'Tree agent roll-out')
    # run_tree_agent(new_tree, env, start_state=state, render=True)
    # tree_pool.append(new_tree)


# train trees
# for each of the N starting states, we have trained n_iter cem agents.
# for each cem agent, train a tree with depth params['maxdepth'].
# these become the pool

# tree_pool = []
# for i in range(len(sample_states)):
#     print(f'Training trees for state {sample_states[i]}')
#     for agent in cem_agents[i]:
#         history = generate_history(env, agent, sample_states[i], num_iter=4000)
#         new_tree = RLTree(history, params['max_depth'])
#         tree_pool.append(new_tree)
#         print(new_tree)

# for i in range(5):
#     rnd_idx = np.random.randint(len(tree_pool))
#     rnd_tree = tree_pool[rnd_idx]
#     for state in sample_states:
#         print(f'running tree {i} from state: {state}')
#         run_tree_agent(rnd_tree, env, start_state=state, render=True)


# initialize forest
F = StochasticArbor()
# initialize weights
w = np.ones(params['N']+1)  # additional 1 is for default start state

while F.size() < 1:
# params['forest_size']:

    min_sum_weights = np.inf
    min_sum_tree = np.inf  # index of tree with least sum of weights w/ loss
    profit_of_min_sum_tree = -np.inf
    min_sum_tree_loss_indicators = np.array([])
    min_sum_tree_loss = np.inf
    min_sum_tree_rewards = np.array([])
    for i in range(len(tree_pool)):
        # for each tree, calculate its return for every starting state
        tree_rewards = np.array([])
        for state in sample_states:
            state_reward = run_tree_agent(tree_pool[i], env, start_state=state, render=True)
            tree_rewards = np.append(tree_rewards, state_reward)
        print(f'Tree {i} rewards: {tree_rewards}')
        print(f'Oracle rewards: {oracle_rollout_rewards}')
        # calculate loss
        # loss = oracle_rollout_rewards - tree_rewards
        # weighted loss
        loss = w* (oracle_rollout_rewards - tree_rewards)
        positive_loss_indicators = np.array([1 if x > 0 else 0 for x in loss])
        negative_loss_indicators = np.array([1 if x <= 0 else 0 for x in loss])
        sum_w = np.sum(w*positive_loss_indicators)
        rewards_total = np.sum(tree_rewards)
        print(f"Tree {i} weight sum: {sum_w}, loss: {loss},  total rewards: {rewards_total}")
        if sum_w < min_sum_weights:
            min_sum_tree = i
            min_sum_weights = sum_w
            profit_of_min_sum_tree = rewards_total
            min_sum_tree_loss_indicators = positive_loss_indicators
            min_sum_tree_loss = loss
            min_sum_tree_rewards = tree_rewards
        elif sum_w == min_sum_weights:
            # tie goes to the tree with the most profit
            if rewards_total > profit_of_min_sum_tree:
                min_sum_weights = sum_w
                min_sum_tree = i
                profit_of_min_sum_tree = rewards_total
                min_sum_tree_loss_indicators = positive_loss_indicators
                min_sum_tree_loss = loss
                min_sum_tree_rewards = tree_rewards
    print(f"tree with best score: {min_sum_tree}, sum of weights: {min_sum_weights}, net profit: {profit_of_min_sum_tree}")
    T = tree_pool.pop(min_sum_tree)
    # calculate T's error:
    # should this be weighted error?
    error = (oracle_rollout_rewards - min_sum_tree_rewards) / oracle_rollout_rewards
    # adjust
    w = w * np.exp(error)


    # error_numerator = np.sum(w * min_sum_tree_loss * min_sum_tree_loss_indicators)
    # error_denominator = np.sum(w * oracle_rollout_rewards)
    # error = error_numerator / error_denominator
    # assert( 0 <= error <= 1)
    # # calculate update coefficient alpha
    # if error == 0:
    #     alpha = np.inf
    # else:
    #     alpha = .5 * np.log2((1-error)/error)
    #     print(alpha)

    #     w_loss = w * min_sum_tree_loss_indicators * np.exp(alpha)
    #     w_gain = w * np.array([1 if x==0 else 0 for x in min_sum_tree_loss_indicators]) * np.exp(-alpha)
    #     w = w_loss + w_gain
    # print(f'w_loss: {w_loss}')
    # print(f'w_gain: {w_gain}')
    print(f'updated w: {w}')
    # calculate T's beta value using sigmoid function
    # x = sum(tree_rewards - oracle_rollout_rewards)
    # print(x)
    # T.beta = 1 / (1 + np.exp(-x))
    # print(f'Beta: {T.beta}')
    # add tree to forest
    # calculate beta as sum of T's rewards / sum of oracle's rewards
    T.beta = sum(tree_rewards) / sum(oracle_rollout_rewards)
    print(f'beta: {T.beta}')
    F.add_tree(T)








'''
Options
=======

Generate a tree for every sample history created
generate multiple trees per history
Using samples from each history to create multiple trees
Generate trees from the entire search space
'''
env.close()

# if __name__ == '__main__':
#     # kde_contour(oracle_history[:, 0], oracle_history[:, 1])
#     # print(sample_states)
#     history, reward = sarsa_agent_rollout(Q, np.array([0.03081904,  1.18915886, -0.03475498, -0.28912238]) ,render=True)
#
#     print(f'Total reward: {reward}')
#     print(history)
