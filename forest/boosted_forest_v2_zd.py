from sklearn.neighbors import KernelDensity
from agent_rollouts import *
from train_cem_agent import *
from stochastic_arbor import *

'''
Here the oracle is the zero distance agent, which has an average reward of 10000 out of 10000
over 100 episodes
'''


oracle_dir = '/Users/ironchefnate/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/CSCI_699_HRI/project/code/robolog/GA/cem/augmented-reward/zero-distance'
oracle_file = 'agent-0019.pkl'
oracle_history_file = 'zero-distance-agent-history-10k.csv'
workdir = '/Users/ironchefnate/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/experiments/boosted_forest_v2_zd'
with open(path.join(oracle_dir, oracle_file), 'rb') as f:
    cem_agent = pickle.load(f)

params = dict(N=300, max_depth=4, target="CartPole-v2", forest_size=10, bandwidth=.05)

env = gym.make(params['target'])

# calculate distribution of oracle states and sample N points
oracle_history = np.genfromtxt(path.join(oracle_dir, oracle_history_file), delimiter=',')
oracle_history = oracle_history[:, :-1]
kde = KernelDensity(kernel='gaussian', bandwidth=params['bandwidth']).fit(oracle_history)
print(f'calculating KDE with bandwidth {params["bandwidth"]}')
sample_states = kde.sample(params['N'])


# add usual starting state for gym env
# usual_start_state = env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
# sample_states = np.vstack((sample_states, usual_start_state))
num_states = params['N']

# do roll-out of oracle for every sample state and record history and rewards
oracle_rewards = np.array([])
for state in sample_states:
    reward = run_cem_agent(cem_agent, env, state, render=False)
    oracle_rewards = np.append(oracle_rewards, reward)
print(f'Oracle rewards: {oracle_rewards}')
oracle_rewards_ind = oracle_rewards.argsort()[::-1]
# print(oracle_rewards_ind)
# create pool of candidate trees using cem agents as models
# use only the last cem agent created
tree_pool = []
for state in sample_states:
    thetas, means = train_agents(env, start_state=state, n_iter=20, n_steps=500, render=False)
    theta = thetas[-1]
    agent = BinaryActionLinearPolicy(theta)
    history = generate_history(env, agent, start_state=state, num_iter=4000, render=False)
    new_tree = RLTree(history, params['max_depth'])
    # run_tree_agent(new_tree, env, start_state=state, render=True)
    tree_pool.append(new_tree)
num_trees = len(tree_pool)

print("Initializing forest...")
F = StochasticArbor()
weights = np.ones(num_states)

while F.size() < params['forest_size']:
    '''
    Iterate through the pool of trees.  
    For each tree, calculate its returns starting from every state in sample_states.
    Calculate the difference between the tree's returns and the oracle's returns.
    Sum the weights corresponding to states where the oracle had a higher return.
    The tree that has the minimal sum will be added to the forest.
    Calculate its beta value.
    Calculate the update coefficient alpha and update the weights.
    '''
    min_tree_idx = 0
    min_sum = np.inf
    min_tree_rewards = np.ones_like(sample_states) * -np.inf
    min_tree_loss_indicators = np.ones_like(sample_states)

    for i in range(num_trees):
        tree_rewards = np.array([])
        for state in sample_states:
            state_reward = run_tree_agent(tree_pool[i], env, start_state=state, render=False)
            tree_rewards = np.append(tree_rewards, state_reward)
        loss = oracle_rewards - tree_rewards
        loss_indicators = np.array([1 if x > 0 else 0 for x in loss])
        sum_weights = np.sum(weights * loss_indicators)
        if sum_weights < min_sum or (sum_weights == min_sum and np.sum(tree_rewards) > np.sum(min_tree_rewards)):
            min_tree_idx = i
            min_sum = sum_weights
            min_tree_loss_indicators = loss_indicators
            min_tree_rewards = tree_rewards

    T = tree_pool.pop(min_tree_idx)
    num_trees -= 1
    error = (oracle_rewards - min_tree_rewards) / oracle_rewards
    weights = weights * np.exp(error)
    T.beta = np.sum(min_tree_rewards) / np.sum(oracle_rewards)
    F.add_tree(T)
    print(f'Added tree {len(F.trees)} to forest,  Beta={T.beta}.')
    #print(f'Sum of weights: {min_sum}, rewards: {min_tree_rewards}, beta: {T.beta}')
    #print(f'Updated weights : {weights}')
    print('Forest demo...')
    env.close()
    # env = gym.make('CartPole-v2')
    demo_rewards = np.array([])
    for i in range(100):
        if i == 99: render = True
        else: render = False
        reward = run_forest_agent(F, env, render=render)
        demo_rewards = np.append(demo_rewards, reward)
    print(f'Forest with {len(F.trees)} trees, avg reward: {np.mean(demo_rewards)}, max/min: '
          f'{np.max(demo_rewards)},{np.min(demo_rewards)}, var: {np.var(demo_rewards)}, '
          f'std: {np.std(demo_rewards)}')
    # forest_state_rewards = np.array([])
    # env = gym.make('CartPole-v1')
    # for state in sample_states:
    #     print(state)
    #     reward = run_forest_agent(F, env, start_state=state, render=True)
    #     print(f'Reward: {reward}')
    #     forest_state_rewards = np.append(forest_state_rewards, reward)
    # if np.sum(forest_state_rewards - oracle_rewards) >= 0:
    #     break
F.print_trees()
# run_forest_agent(F, env, render=True)
env.close()
# save F
filename = f'Forest_depth{params["max_depth"]}_samples{params["N"]}_cp2_bw05_00.pkl'
with open(workdir + '/' + filename, 'wb') as f:
    pickle.dump(F, f)


