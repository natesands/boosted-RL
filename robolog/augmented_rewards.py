"""
convert state action pairs to csv file
store in work direction
run rewards.dl
open rewards file and calculate sum
return sum
"""
import subprocess
import numpy as np
from typing import List
from os import path


def invoke_souffle(dl_file_directory: str, dl_file_name: str):
    result = subprocess.run(['souffle', '-F', dl_file_directory, '-D',
                             dl_file_directory, dl_file_directory + '/' + dl_file_name],
                            capture_output=True, encoding='utf-8')
    print(result)


def read_entry(time: int, file: str):
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if int(line[0]) == time:
            return line
    return None


def write_facts(facts: np.ndarray, write_dir: str, filename: str):
    with open(path.join(write_dir, filename), 'w') as f:
        for fact in facts:
            fact = [str(x) for x in fact]
            fact = '\t'.join(fact) + '\n'
            f.write(fact)


# rewards.csv has format 'time <tab> reward'
def tally_rewards(workdir: str, rewards_csv):
    with open(path.join(workdir, rewards_csv), 'r') as f:
        lines = f.readlines()
    time_reward = []
    for line in lines:
        line = line.strip().split('\t')
        line = [float(s) for s in line]
        time_reward.append(line)
    time_reward = np.array(time_reward)
    reward = sum(time_reward[:, -1])
    return reward


def augmented_reward(state_action_pairs: np.ndarray, workdir: str, rewards_dl_file: str) -> int:
    state_facts = state_action_pairs[:, :-1]
    action_facts = np.concatenate((state_action_pairs[:, 0].reshape(-1, 1), state_action_pairs[:, -1].reshape(-1, 1)),
                                  axis=1)
    write_facts(state_facts, workdir, 'state.facts')
    write_facts(action_facts, workdir, 'action.facts')
    invoke_souffle(workdir, rewards_dl_file)
    aug_reward = tally_rewards(workdir, 'reward.csv')
    return aug_reward


if __name__ == '__main__':
    workdir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/GA/cem/tmp'
    invoke_souffle(workdir, 'test.dl')
    a = [[1, 0, 4, 5, 2, 0],
         [2, -3, -5, 2, 1, 1],
         [3, 0, 0, 40, 0, 1],
         [4, 0, 0, 0, 80, 0]]
    a = np.array(a)
    print(augmented_reward(a, workdir, 'test.dl'))
