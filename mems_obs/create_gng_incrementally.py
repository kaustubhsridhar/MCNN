 

import numpy as np
import collections
import pickle
from neural_gas_helpers import data2gas
from franka_kitchen_helpers import create_gng_incremental_for_frankakitchen
import os 
from collections import defaultdict
import time
import argparse 
from tqdm import tqdm
from copy import deepcopy

# hyperparameters
parser = argparse.ArgumentParser(description='Parse constants.')
parser.add_argument('--name', default='halfcheetah-medium-v2', type=str, help="")
parser.add_argument('--gng_epochs', default=1, type=int, help='num epochs for gng')
parser.add_argument('--num_memories_frac', type=float, default=0.05)
args = parser.parse_args()

# setup
folder = f'mems_obs/memories'
os.makedirs(folder, exist_ok=True)

# for kitchen data only since it has a different dataset format
if args.name == 'kitchen':
    create_gng_incremental_for_frankakitchen(folder, args)
    exit()

# load top_paths
with open(f'data/top_paths.pkl', 'rb') as f:
    top_paths = pickle.load(f)

# load paths
if 'carla' in args.name:
    with open(f'data/datasets/{args.name}_embeddings.pkl', 'rb') as f:
        all_paths = pickle.load(f)
else:
    with open(f'data/datasets/{args.name}.pkl', 'rb') as f:
        all_paths = pickle.load(f)
num_total_points = np.sum([p['rewards'].shape[0] for p in all_paths])

# incremental neural gas creation
prev_graph = None
for chosen_percentage in [1.0]: # 0.1, 0.2, 0.5, 1.0
    # full name
    full_name = f'{args.name}-{chosen_percentage}'
    
    # load train paths
    if chosen_percentage == 1.0:
        train_paths = deepcopy(all_paths)
    else:
        train_path_indices = top_paths[full_name]
        train_paths = [all_paths[i] for i in train_path_indices]
    
    num_train_points = np.sum([p['rewards'].shape[0] for p in train_paths])
    num_memories = int(num_train_points * args.num_memories_frac)

    # save name of gng file
    gng_name = f'{folder}/{full_name}/memories_{args.num_memories_frac}_frac.pkl'
    os.makedirs(f'{folder}/{full_name}', exist_ok=True)
    t0 = time.time()

    # create gng if it doesn't exist
    if not os.path.exists(gng_name):
        # collect paths
        if 'carla' in args.name:
            all_observations = np.concatenate([p['embeddings'] for p in train_paths])
        else:
            all_observations = np.concatenate([p['observations'] for p in train_paths])
        all_actions = np.concatenate([p['actions'] for p in train_paths])
        print(f'{args.name} {all_observations.shape=}, {all_actions.shape=}')

        # create neural gas
        gng = data2gas(states=all_observations, max_memories=num_memories, gng_epochs=args.gng_epochs)

        with open(f'{gng_name}', 'wb') as f:
            pickle.dump(gng, f)
        print(f'==> for creating gng --- {full_name} {args.num_memories_frac=}, i.e. {num_memories=}/{num_train_points} duration={time.time()-t0}')
    else:
        print(f'==> already exisiting gng --- {full_name} {args.num_memories_frac=}, i.e. {num_memories=}/{num_train_points} duration={time.time()-t0}')


