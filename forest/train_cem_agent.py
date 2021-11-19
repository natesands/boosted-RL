import gym
from gym import wrappers, logger
import pickle
import json, sys, os
from os import path
import numpy as np
from typing import List, Tuple


class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

    def __repr__(self):
        return np.append(self.w, self.b).__repr__()

def cem(env, start_state, f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0, n_steps=200, n_sessions=1, render=False):
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(env, start_state, th, n_steps, n_sessions, render) for th in
                       ths])  # f will will run scenario and return a reward
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)  # average the elite thetas
        th_std = elite_ths.std(axis=0)  # new std
        yield {'ys': ys, 'theta_mean': th_mean, 'y_mean': ys.mean()}


def do_rollout(agent, env, num_steps, num_sessions, start_state: np.ndarray, render=False):
    rewards = []
    time_total = 0
    for _ in range(num_sessions):
        total_rew = 0
        ob = env.reset_to_state(start_state)  # resets environment and returns initial observation
        for t in range(num_steps):
            a = agent.act(ob)
            ob, reward, done, _info = env.step(a)
            total_rew += reward
            if render and t % 3 == 0: env.render()
            if done: break
        rewards.append(total_rew)
        time_total += t + 1  # is this even necessary?
    return np.mean(rewards), time_total / num_sessions


def noisy_evaluation(env, start_state, theta: np.ndarray, n_steps, n_sessions, render=False):
    agent = BinaryActionLinearPolicy(theta)
    rew, T = do_rollout(agent, env, n_steps, n_sessions, start_state, render)
    return rew


def train_agents(env, start_state=None, th_mean=np.zeros(5), initial_std=1.0, n_iter=20, n_sessions=1,
                 n_steps=200,
                 batch_size=25, elite_frac=.2, render=False) -> Tuple[List[np.ndarray], List[float]]:
    if start_state is None:
        start_state = env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    new_agents = cem(env, start_state, noisy_evaluation, th_mean, batch_size, n_iter, elite_frac, initial_std, n_steps,
                     n_sessions, render)
    thetas = []  # agents are in order of generation
    means = []
    for agent in new_agents:
        thetas.append(agent['theta_mean'])
        means.append(agent['y_mean'])
    return thetas, means

def generate_history(env, agent: BinaryActionLinearPolicy, start_state: np.ndarray,
                     num_iter: int, add_noise=False, render=False) -> np.ndarray:
    # returns 4 coordinate state + action as numpy array
    observation = env.reset_to_state(start_state)
    state_action_pairs = []
    for _ in range(num_iter):
        if render:
            env.render()
        action = agent.act(observation)
        state_action_pairs.append(list(observation) + [action])
        observation, reward, done, info = env.step(action)
        if done:
            if add_noise:
                start_state += env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            observation = env.reset_to_state(start_state)
    return np.array(state_action_pairs)

if __name__ == '__main__':
    target = 'CartPole-v1'
    env = gym.make(target)
    env.seed(0)
    np.random.seed(0)
    start_state = env.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    thetas, means = train_agents(env, np.array([-1,0,0,0]), n_steps=500, render=False)
    print(thetas)
    print(means)
    a = BinaryActionLinearPolicy(thetas[5])
    history = generate_history(env, a, start_state, 1000, add_noise=True, render=True)
    print(history)


    # for i in range(len(thetas)):
    #     a = BinaryActionLinearPolicy(thetas[i])
    #     m = means[i]
    #     print(f'agent: {thetas[i]}, mean reward: {m}')
    #     do_rollout(a, new_env.env, 500, 1, np.array([1,0,0,0]), render=True)
