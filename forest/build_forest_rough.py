from stochastic_arbor import *
import gym
from gym import wrappers
from train_cem_agent import *

target = "CartPole-v2"
work_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/forest/workdir'


# record video or no -----------------------
def record_video(i: int) -> bool:  # returns True if episode i should be recorded
    return False


outdir = path.join(work_dir, 'videos')


def tree_state_rewards(tree: RLTree, states: np.ndarray):
    env = gym.make(target)
    # env = wrappers.Monitor(env, outdir, video_callable=record_video, force=True)
    state_rewards = []
    state_count = 0
    for state in states:
        state_count += 1
        observation = env.reset_to_state(state)
        total_reward = 0
        for _ in range(10000):  # upper limit for session length
            total_reward += 1
            # env.render()
            action = tree.act(observation.reshape(1, -1))
            observation, reward, done, info = env.step(action)
            if done:
                print(f'State {state_count}: {total_reward} steps\n')
                state_rewards.append(np.concatenate((state, [total_reward])))
                total_reward = 0
                break
    env.close()
    return np.array(state_rewards)


# --------------------------------------------------------------------
# initialize weights
data_size = 3999
weights = np.ones(data_size) / data_size

# oracle (sarsa agent 3) state-reward history (4000 samples) ----------
oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
data_file = 'state_rewards.csv'
oracle_state_rewards = np.genfromtxt(path.join(oracle_dir, data_file), delimiter=",")

# train first tree to represent oracle --------------------------------
data_file = 'train_data.csv'
train_data = np.genfromtxt(path.join(oracle_dir, data_file), delimiter=",")

T1 = RLTree(train_data, max_depth=5)
print(f'Initialized Tree (max depth = {T1.max_depth}) with rules:\n')
for rule in T1.get_rules():
    print(rule)

# sample_state_rewards = np.array([
#     [1.748892049245034, 0.5874897887518551, 0.10521271415342826, -1.5516452754715155, 141],
#     [0.035524680836036104, -0.47823223465729936, -0.07761819265747755, -0.662874409620759, 106],
#     [0.08436992064339544, 0.8076704451067818, -0.0380641441820766, -0.39336760762885037, 441],
#     [1.9549072743730238, -0.30553700597470046, -0.0022330383191002667, -0.9746085717030586, 63]
# ])
sample_state_rewards = oracle_state_rewards
oracle_states = sample_state_rewards[:, :-1]
oracle_rewards = sample_state_rewards[:,-1]
T_state_rewards = tree_state_rewards(T1, sample_state_rewards[:, :-1])
# calculate loss across all samples
loss = oracle_rewards - T_state_rewards[:,-1]
loss = np.maximum(np.zeros_like(loss), loss)
loss_indicators = np.array([1 if x > 0 else 0 for x in loss])
# need vector of 1's where loss is positive
print(loss)
print(loss_indicators)

error = 0 if sum(loss_indicators) == 0 else sum((weights * loss) * loss_indicators) / sum((weights*oracle_rewards)*loss_indicators)
print(error)


# calculate error
# error = sum(weights*loss)/sum(weights*oracle_rewards)

alpha = 0 if error == 0 else np.log2((1 - error)/error)
print(alpha)
gain_indicators = [1 if x == 0 else 0 for x in loss_indicators]
beta = sum(gain_indicators) / data_size
T1.beta = beta
# update weights
print(weights)
weights = weights * np.exp(-alpha * loss_indicators)
print(weights)

weights_ind = weights.argsort()[::-1]
print(weights_ind)
next_sample_ind = weights_ind[0]
next_sample_to_address = oracle_states[next_sample_ind]
print(next_sample_to_address)
F = StochasticArbor()
F.add_tree(T1)

# train a new agent
env = gym.make(target)

thetas, means = train_agents(env, next_sample_to_address, n_steps=500)
new_agent_theta = thetas[-1]
print(new_agent_theta)
new_agent = BinaryActionLinearPolicy(new_agent_theta)

new_history = generate_history(env, new_agent, next_sample_to_address, num_iter=4000, add_noise=False, render=False)
print(new_history)

T2 = RLTree(new_history, max_depth=5)
print(f'Initialized Tree (max depth = {T1.max_depth}) with rules:\n')
for rule in T1.get_rules():
    print(rule)
env.close()

T_state_rewards = tree_state_rewards(T2, sample_state_rewards[:, :-1])

loss = oracle_rewards - T_state_rewards[:,-1]
loss = np.maximum(np.zeros_like(loss), loss)
loss_indicators = np.array([1 if x > 0 else 0 for x in loss])
# need vector of 1's where loss is positive
print(loss)
print(loss_indicators)

error = 0 if sum(loss_indicators) == 0 else sum((weights * loss) * loss_indicators) / sum((weights*oracle_rewards)*loss_indicators)
print(error)


# calculate error
# error = sum(weights*loss)/sum(weights*oracle_rewards)

alpha = 0 if error == 0 else np.log2((1 - error)/error)
print(alpha)
gain_indicators = [1 if x == 0 else 0 for x in loss_indicators]
beta = sum(gain_indicators) / data_size
T2.beta = beta

print(weights)
weights = weights * np.exp(-alpha * loss_indicators)
print(weights)

weights_ind = weights.argsort()[::-1]
print(weights_ind)
next_sample_ind = weights_ind[0]
next_sample_to_address = oracle_states[next_sample_ind]
print(next_sample_to_address)

F.add_tree(T2)

# train a new agent
env = gym.make(target)

thetas, means = train_agents(env, next_sample_to_address, n_steps=500)
new_agent_theta = thetas[-1]
print(new_agent_theta)
new_agent = BinaryActionLinearPolicy(new_agent_theta)

new_history = generate_history(env, new_agent, next_sample_to_address, num_iter=4000, add_noise=False, render=False)
print(new_history)

T3 = RLTree(new_history, max_depth=5)
print(f'Initialized Tree (max depth = {T1.max_depth}) with rules:\n')
for rule in T1.get_rules():
    print(rule)
env.close()

T_state_rewards = tree_state_rewards(T2, sample_state_rewards[:, :-1])

loss = oracle_rewards - T_state_rewards[:,-1]
loss = np.maximum(np.zeros_like(loss), loss)
loss_indicators = np.array([1 if x > 0 else 0 for x in loss])
# need vector of 1's where loss is positive
print(loss)
print(loss_indicators)

error = 0 if sum(loss_indicators) == 0 else sum((weights * loss) * loss_indicators) / sum((weights*oracle_rewards)*loss_indicators)
print(error)


# calculate error
# error = sum(weights*loss)/sum(weights*oracle_rewards)

alpha = 0 if error == 0 else np.log2((1 - error)/error)
print(alpha)
gain_indicators = [1 if x == 0 else 0 for x in loss_indicators]
beta = sum(gain_indicators) / data_size
T3.beta = beta

print(weights)
weights = weights * np.exp(-alpha * loss_indicators)
print(weights)

weights_ind = weights.argsort()[::-1]
print(weights_ind)
next_sample_ind = weights_ind[0]
next_sample_to_address = oracle_states[next_sample_ind]
print(next_sample_to_address)

F.add_tree(T3)

for tree in F.trees:
    print(tree.beta)
# [x+1 if x >= 45 else x+5 for x in l]
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
    return i < 5

outdir = path.join(work_dir, 'videos')
# initialize gym environment -------------------
env = gym.make("CartPole-v2")
env.seed(0)
env = wrappers.Monitor(env, outdir, video_callable=record_video, force=True)
observation = env.reset()
state_action_pairs = []
t = 0
for _ in range(5000):
    t += 1
    env.render()
    # action = T_oracle.act(observation.reshape(1,-1))
    action = F.vote(observation.reshape(1, -1))
    # state_action_pairs.append(list(observation) + [action])
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        print(f'{t} steps\n')
        t = 0
# record_session(state_action_pairs)
env.close()


# with open(path.join(oracle_dir, 'train_data.csv'), 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip().split(',')
#         line = [float(s) for s in line]
#         train_data.append(line)
# train_data = np.array(train_data)
