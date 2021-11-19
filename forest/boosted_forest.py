from sklearn.neighbors import KernelDensity
from agent_rollouts import *
from train_cem_agent import *
from stochastic_arbor import *

oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
oracle_file = 'cp-sarsa-agent-Q-04.pkl'
oracle_history_file = 'history_10000_cpv1.csv'
with open(path.join(oracle_dir, oracle_file), 'rb') as f:
    Q = pickle.load(f)

params = dict(N=10, max_depth=3, target="CartPole-v1", forest_size=10)

env = gym.make(params['target'])

# calculate distribution of oracle states and sample N points
oracle_history = np.genfromtxt(path.join(oracle_dir, oracle_history_file), delimiter=',')
oracle_history = oracle_history[:, :-1]
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(oracle_history)
sample_states = kde.sample(params['N'])
num_states = params['N'] + 1

# add usual starting state for gym env
usual_start_state = env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
sample_states = np.vstack((sample_states, usual_start_state))

# do roll-out of oracle for every sample state and record history and rewards
oracle_histories = []
oracle_rewards = np.array([])
for state in sample_states:
    history, reward = sarsa_agent_rollout(Q, env, state, render=False)
    oracle_histories.append(history)
    oracle_rewards = np.append(oracle_rewards, reward)
print(f'Oracle rewards: {oracle_rewards}')

# create pool of candidate trees using cem agents as models
tree_pool = []
for state in sample_states:
    thetas, means = train_agents(env, start_state=state, n_iter=20, n_steps=500, render=False)
    for theta in thetas:
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
    print(f'Added tree {min_tree_idx} to forest:')
    print(f'Sum of weights: {min_sum}, rewards: {min_tree_rewards}, beta: {T.beta}')
    print(f'Updated weights : {weights}')
    print('Forest demo...')
    forest_state_rewards = np.array([])
    for state in sample_states:
        print(state)
        reward = run_forest_agent(F, env, start_state=state, render=True)
        print(f'Reward: {reward}')
        forest_state_rewards = np.append(forest_state_rewards, reward)
    if np.sum(forest_state_rewards - oracle_rewards) >= 0:
        break
F.print_trees()
env.close()



