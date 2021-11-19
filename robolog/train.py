import gym
from gym import wrappers, logger
import pickle
import json, sys, os
from os import path
import argparse
from augmented_rewards import *


class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

outdir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/GA/cem/augmented-reward/zero-distance'
workdir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/GA/cem/workdir'
rewards_dl_file = 'rewards.dl'
# cem is called
# initializes batch size array of thetas
# over n_iter rounds it calls f on each thetha
# based on f's return, it orders them and picks the top n_elite
# these elites are averaged... to get a new theta
# from this theta, generate a new batch of thetas

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.5):
    print(th_mean)
    th_mean = th_mean + np.random.normal(0, .1, len(th_mean))
    print(th_mean)
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])  # f will will run scenario and return a reward
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)  # average the elite thetas
        th_std = elite_ths.std(axis=0)  # new std
        yield {'ys': ys, 'theta_mean': th_mean, 'y_mean': ys.mean()}


def do_rollout(agent, env, num_steps, num_sessions, render=False):
    rewards = []
    time_total = 0
    for _ in range(num_sessions):
        total_rew = 0
        ob = env.reset()  # resets environment and returns initial observation
        state_action = []
        for t in range(num_steps):
            # ob = mod_ob(ob)  <------- add say acceleration here if necess
            a = agent.act(ob)
            state_action.append(np.concatenate(([t], ob,[a])))
            ob, reward, done, _info = env.step(a)
            total_rew += reward
            if render and t % 3 == 0: env.render()
            if done: break
        # aug_reward = augmented_reward(np.array(state_action), workdir, rewards_dl_file)
        # total_rew += aug_reward
        # total_rew = aug_reward
        # calculate additional reward based on state_action pairs in session
        # add reward to rewards
        rewards.append(total_rew)
        time_total += t + 1   # is this even necessary?
    return np.mean(rewards), time_total/num_sessions

def clean_work_dir(work_dir:str):
    for file in os.listdir(work_dir):
        if file.endswith('.facts'):
            os.remove(path.join(work_dir, file))


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default="CartPole-v1")
    args = parser.parse_args()

    clean_work_dir(workdir)

    env = gym.make(args.target)
    env.seed(15)
    np.random.seed(15)
    params = dict(n_iter=20, batch_size=25, elite_frac=0.2)
    num_steps = 500
    num_sessions = 1

    def record_video(i: int) -> bool:  # returns True if episode i should be recorded
        return False


    # outdir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/GA'
    env = wrappers.Monitor(env, outdir, video_callable=record_video, force=True)


    # Prepare snapshotting
    # -------------------------------------
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)


    def writepickle(fname, agent):
        with open(path.join(outdir, fname), 'wb') as fh:
            pickle.dump(agent, fh, -1)


    info = {}
    info['params'] = params
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id


    # -------------------------------------

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps, num_sessions)
        return rew


    # Train the agent and snapshot each stage
    for (i, iterdata) in enumerate(
            cem(noisy_evaluation, np.zeros(env.observation_space.shape[0] + 1), **params)):
        print('Iteration %2i.  Episode mean reward : %7.3f' % (i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        #if args.display:
        if True:
            do_rollout(agent, env, 1000, 1,  render=True)
        pickled = pickle.dumps(agent, -1)
        writepickle('agent-%.4i.pkl' % i, agent)

    writefile('info.json', json.dumps(info))
    clean_work_dir(workdir)
    env.close()
