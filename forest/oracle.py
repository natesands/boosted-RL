from os import path
import numpy as np


class Oracle:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.states = []
        self.rewards = []

    def load_history(self, file_name: str):
        with open(path.join(self.work_dir, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                start_state = [float(s) for s in np.array(line[:-1])]
                reward = int(line[-1])
                self.states.append(start_state)
                self.rewards.append(reward)

    def total_reward(self) -> int:
        return sum(self.rewards)

    def get_reward(self, state: np.ndarray) -> float:
        i = self.states.index(state)
        return self.rewards[i]

    def avg_reward(self) -> float:
        return self.total_reward() / len(self.states)



if __name__ == '__main__':
    work_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
    history = 'state_rewards.csv'
    O = Oracle(work_dir)
    O.load_history(history)
    print(O.total_reward())
    print(O.states[0])
