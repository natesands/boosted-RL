# from export import *
import numpy as np
from typing import List
from sklearn import tree
from os import path
from export import *
from robolog_classes import *


class RLTree:
    def __init__(self, train_data: np.ndarray, max_depth=3):
        self.X = train_data[:, :-1]
        self.y = train_data[:, -1].astype(np.int64)
        self.T = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.T.fit(self.X, self.y)
        self.beta = 1
        self.max_depth = max_depth

    def act(self, observation):
        return self.T.predict(observation)[0]
        # note: the value returned by predict is of the form [<prediction>], hence the '[0]'

    def vote(self, observation):
        if self.act(observation) == 1:
            return 1
        else:
            return -1

    def get_rules(self):
        rules = []
        d = export_dict(self.T)
        paths = tree_paths(d)
        for path in paths:
            rules.append(path_to_rule(path))
        return rules

    def __repr__(self):
        return f'RLTree(Beta={self.beta}, max_depth={self.max_depth})'


class StochasticArbor:
    def __init__(self):
        self.trees = []

    def vote(self, observation):
        votes = [tree.beta * tree.vote(observation) for tree in self.trees]
        tally = sum(votes)
        if tally >= 0:
            return 1
        else:
            return 0

    def add_tree(self, tree: RLTree):
        self.trees.append(tree)

    def size(self):
        return len(self.trees)

    def print_trees(self):
        for tree in self.trees:
            print(tree)
            print(tree.get_rules())


# ----------------------------------------

num_features = 4
op_map = {'gt': '>', 'leq': '<='}


def tree_paths(d: dict) -> list:
    paths = []

    def tree_paths_aux(d: dict, path_so_far):
        if d["feature"] is None:
            rule = path_so_far + d["value"]
            paths.append(rule)
        else:
            assert (d["feature"] is not None)
            lpath = path_so_far + [(d["feature"], 'leq', d["threshold"])]
            tree_paths_aux(d["left"], lpath)
            rpath = path_so_far + [(d["feature"], 'gt', d["threshold"])]
            tree_paths_aux(d["right"], rpath)

    tree_paths_aux(d, [])
    return paths


def path_to_rule_str(path: list) -> str:
    assert (type(path[-1]) == list)
    preds = []
    variables = ['_'] * num_features
    for feat, op, bound in path[:-1]:
        variable = 's' + str(feat)
        variables[feat] = variable
        pred = ' '.join([variable, op_map[op], str(bound)])
        preds.append(pred)
    state_pred = 'state(t, ' + ', '.join(variables) + ')'
    leaf_counts = path[-1]
    max_val = max(leaf_counts)
    response = leaf_counts.index(max_val)
    action_pred = 'action(t, ' + str(response) + ')'
    body = ', '.join([state_pred] + preds)
    return action_pred + ' :- ' + body


def path_to_rule(path: list) -> Rule:
    assert (type(path[-1]) == list)
    preds = []
    variables = ['_'] * num_features
    for feat, op, bound in path[:-1]:
        variable = 's' + str(feat)
        variables[feat] = variable
        pred = Inequality(feat, op, bound)
        preds.append(pred)
    variables = ['t'] + variables  # add time variable
    state_pred = Predicate('state', variables)
    preds.append(state_pred)
    leaf_counts = path[-1]
    response = leaf_counts.index(max(leaf_counts))  # 0 or 1
    action_pred = Predicate('action', ['t', str(response)])
    return Rule(action_pred, preds)


if __name__ == '__main__':
    oracle_dir = '/Users/ironchefnate/iCloud/Documents/USC/CSCI_699_HRI/project/code/robolog/boost/oracles/sarsa/agent-00'
    train_data = []
    with open(path.join(oracle_dir, 'train_data.csv'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            line = [float(s) for s in line]
            train_data.append(line)

    train_data = np.array(train_data)
    print(train_data)
    a_tree = RLTree(train_data, max_depth=2)
    ob = np.array([-1, 1, .01, 2]).reshape(1, -1)
    print(a_tree.act(ob))
    tree_rules = a_tree.get_rules()
    for rule in tree_rules:
        print(rule)
