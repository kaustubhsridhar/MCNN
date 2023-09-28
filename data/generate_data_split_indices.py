import gym
import numpy as np

import collections
import pickle, os

import d4rl

if os.path.exists('data/splits.pkl'):
    with open('data/splits.pkl', 'rb') as f:
        splits = pickle.load(f)
        print(f'loaded splits from splits.pkl')
else:
    splits = {}
np.random.seed(0)

# print(f'full_name \t len(train_indices) \t len(test_indices) \t intersection(train_indices, test_indices)')
for env_name in ['halfcheetah', 'hopper', 'walker2d']:
    for dataset_type in ['random', 'medium', 'medium-replay', 'expert', 'medium-expert']:
        name = f'{env_name}-{dataset_type}-v2'

        with open(f'data/datasets/{name}.pkl', 'rb') as f:
            paths = pickle.load(f)
        num_paths = len(paths)

        indices = [path_idx for path_idx in range(num_paths)]
        np.random.shuffle(indices)

        test_size = 0.05
        for train_size in [0.1, 0.2, 0.5, 0.95]:
            train_indices = indices[ : int(train_size * len(indices))]
            test_indices = indices[-int(test_size * len(indices)) : ]

            full_name = f'{name}_{round(train_size, 2)}_train'
            if full_name not in splits:
                splits[full_name] = {
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                }

            print(full_name, len(train_indices))

        with open('data/splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        