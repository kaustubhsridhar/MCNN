import gym
import numpy as np

import collections
import pickle, os
from copy import deepcopy

import d4rl

top_paths = {}

# print(f'full_name \t len(train_indices) \t len(test_indices) \t intersection(train_indices, test_indices)')
for env_name in ['halfcheetah', 'hopper', 'walker2d']:
    for dataset_type in ['expert', 'medium-expert', 'medium', 'medium-replay', 'random']:
        name = f'{env_name}-{dataset_type}-v2'

        with open(f'data/datasets/{name}.pkl', 'rb') as f:
            paths = pickle.load(f)
        num_paths = len(paths)

        path_rewards = []
        for path_idx, path in enumerate(paths):
            sum_of_rewards = np.sum(path['rewards'])
            path_rewards.append((path_idx, sum_of_rewards))
        path_rewards = sorted(path_rewards, key=lambda x: x[1], reverse=True)
        
        for chosen_percentage in [0.1, 0.2, 0.5]:
            num_subset_paths = int(num_paths * chosen_percentage)
            full_name = f'{name}-{chosen_percentage}'
            top_paths[full_name] = [tup[0] for tup in path_rewards[:num_subset_paths]]

with open('data/top_paths.pkl', 'wb') as f:
    pickle.dump(top_paths, f)
        