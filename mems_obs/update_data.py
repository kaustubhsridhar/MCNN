import numpy as np
import collections
import pickle
from neural_gas_helpers import data2gas
import os 
from collections import defaultdict
import time
import argparse 
import torch 
from tqdm import tqdm
from copy import deepcopy

# # assuming tensor1 is subset of tensor2, find indices of tensor1's rows in tensor2
# def find_indices(tensor1, tensor2):
# 	indices = []
# 	for idx1, row1 in enumerate(tensor1):
# 		# Check if the row1 exists in tensor2
# 		matches = torch.all(torch.eq(tensor2, row1), dim=1)
# 		# If there's a match, find the index of the matching row in tensor2
# 		if matches.any():
# 			index = torch.where(matches)[0].item()
# 			indices.append(index)
# 		else:
# 			print(f"Row {idx1} in tensor1 not found in tensor2")
# 	return indices

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
parser = argparse.ArgumentParser(description='Parse constants.')
parser.add_argument('--name', default='halfcheetah-medium-v2', type=str, help="")
parser.add_argument('--gng_epochs', default=1, type=int, help='num epochs for gng')
parser.add_argument('--num_memories_frac', type=float, default=0.05) 
args = parser.parse_args()
print(f'\n\n\n\n')

# setup
folder = f'mems_obs/updated_datasets'
os.makedirs(folder, exist_ok=True)

# for kitchen data only since it has a different dataset format
if args.name == 'kitchen':
    # loadd all observations and actions
    all_observations = np.load(f'diffusion_policy/data/{args.name}/all_observations.npy')
    all_observations_tensor = deepcopy(all_observations)
    all_observations_tensor = torch.from_numpy(all_observations_tensor).float().to(device)
    all_actions = np.load(f'diffusion_policy/data/{args.name}/all_actions.npy')
    # load all node weights
    with open(f'mems_obs/memories/{args.name}/memories_{args.num_memories_frac}_frac.pkl', 'rb') as f:
        gng = pickle.load(f)
    node_weights = []
    for n in gng.graph.edges_per_node.keys():
        node_weights.append(n.weight)
    node_weights = np.array(node_weights)
    node_weights = torch.from_numpy(node_weights).float().to(device)
    # find memories (aka points in train data closest to node_weights)
    t0 = time.time()
    memories = []
    memories_actions = []
    for w in node_weights:
        dists = torch.cdist(w, all_observations_tensor)
        min_dists, nearest_point = dists.min(dim=1)
        nearest_point = nearest_point.item()
        memories.append( all_observations[nearest_point] )
        memories_actions.append( all_actions[nearest_point] )
    memories = np.array(memories)
    memories_actions = np.array(memories_actions)
    print(f'{memories.shape=}, {memories_actions.shape=}')
    memories = torch.from_numpy(memories).float().to(device)
    memories_actions = torch.from_numpy(memories_actions).float().to(device)
    print(f'Finding memories took {time.time() - t0} seconds')
    # Get memory and memory action for every observation/row in the dataset
    mem_observations = []
    mem_actions = []
    batch_size = 1000
    for start in tqdm(range(0, len(all_observations), batch_size)):
        obs = all_observations[start:start+batch_size]
        obs = torch.from_numpy(obs).float().to(device)
        
        # get nearest memories
        dists = torch.cdist(memories, obs)
        _, nearest_memories = dists.min(dim=0)

        mem_observations.append( memories[nearest_memories].cpu().numpy() )
        mem_actions.append( memories_actions[nearest_memories].cpu().numpy() )
    mem_observations = np.concatenate(mem_observations)
    mem_actions = np.concatenate(mem_actions)
    assert mem_observations.shape == all_observations.shape
    assert mem_actions.shape == all_actions.shape
    # save
    os.makedirs(f'{folder}/{args.name}', exist_ok=True)
    updated = {
                'mem_observations': mem_observations,
                'mem_actions': mem_actions,
                'memories_obs': memories.cpu().numpy(), 
                'memories_act': memories_actions.cpu().numpy(),
            }
    with open(f'{folder}/{args.name}/updated_{args.num_memories_frac}_frac.pkl', 'wb') as f:
        pickle.dump(updated, f)
    print(f'Done.')
    exit()

# load paths and top_paths
if 'carla' in args.name:
    with open(f'data/datasets/{args.name}_embeddings.pkl', 'rb') as f:
        all_paths = pickle.load(f)
else:
    with open(f'data/datasets/{args.name}.pkl', 'rb') as f:
        all_paths = pickle.load(f)
num_total_points = np.sum([p['rewards'].shape[0] for p in all_paths])
with open(f'data/top_paths.pkl', 'rb') as f:
    top_paths = pickle.load(f)

for chosen_percentage in [1.0]: # 0.1, 0.2, 0.5, 1.0
    print()
    # full name
    full_name = f'{args.name}-{chosen_percentage}'
    
    # load train paths
    if chosen_percentage == 1.0:
        train_paths = deepcopy(all_paths)
    else:
        train_path_indices = top_paths[full_name]
        train_paths = [all_paths[i] for i in train_path_indices]

    num_train_points = np.sum([p['rewards'].shape[0] for p in train_paths])
    print(f'{full_name=}, {num_train_points=}')

    # save name of updated paths 
    save_name = f'{folder}/{full_name}/updated_{args.num_memories_frac}_frac.pkl'
    os.makedirs(f'{folder}/{full_name}', exist_ok=True)

    if not os.path.exists(save_name):
        # load gng
        gng_name = f'mems_obs/memories/{full_name}/memories_{args.num_memories_frac}_frac.pkl'
        with open(f'{gng_name}', 'rb') as f:
            gng = pickle.load(f)
        node_weights = []
        for n in gng.graph.edges_per_node.keys():
            node_weights.append(n.weight)
        node_weights = np.array(node_weights)
        node_weights = torch.from_numpy(node_weights).float().to(device)

        # paths expanded
        choice = 'embeddings' if 'carla' in args.name else 'observations'
        all_observations = np.concatenate([p[choice] for p in train_paths])
        all_actions = np.concatenate([p['actions'] for p in train_paths])
        all_next_observations = np.concatenate([p[f'next_{choice}'] for p in train_paths])
        all_rewards = np.concatenate([p['rewards'] for p in train_paths])

        all_observations_tensor = deepcopy(all_observations)
        all_observations_tensor = torch.from_numpy(all_observations_tensor).float().to(device)

        # find memories (aka points in train data closest to node_weights)
        t0 = time.time()
        print(f"Finding memories for {len(node_weights)} nodes' weights...")
        memories = []
        memories_actions = []
        memories_next_obs = []
        memories_rewards = []
        for w in node_weights:
            dists = torch.cdist(w, all_observations_tensor)
            min_dists, nearest_point = dists.min(dim=1)
            nearest_point = nearest_point.item()
            memories.append( all_observations[nearest_point] )
            memories_actions.append( all_actions[nearest_point] )
            memories_next_obs.append( all_next_observations[nearest_point] )
            memories_rewards.append( all_rewards[nearest_point] )
        memories = np.array(memories)
        memories_actions = np.array(memories_actions)
        memories_next_obs = np.array(memories_next_obs)
        memories_rewards = np.array(memories_rewards)
        print(f'{memories.shape=}, {memories_actions.shape=}, {memories_next_obs.shape=}, {memories_rewards.shape=}')
        memories = torch.from_numpy(memories).float().to(device)
        memories_actions = torch.from_numpy(memories_actions).float().to(device)
        memories_next_obs = torch.from_numpy(memories_next_obs).float().to(device)
        memories_rewards = torch.from_numpy(memories_rewards).float().to(device)
        print(f'Finding memories took {time.time() - t0} seconds')

        # update train paths
        t0 = time.time()
        print(f'Updating {len(train_paths)} train paths...')
        updated_train_paths = []
        for path in train_paths:
            new_path = deepcopy(path)
            obs = path[choice]
            obs = torch.from_numpy(obs).float().to(device)
            
            # get nearest memories
            dists = torch.cdist(memories, obs)
            min_dists, nearest_memories = dists.min(dim=0)

            new_path['mem_observations'] = memories[nearest_memories].cpu().numpy()
            new_path['mem_actions'] = memories_actions[nearest_memories].cpu().numpy()
            new_path['mem_next_observations'] = memories_next_obs[nearest_memories].cpu().numpy()
            new_path['mem_rewards'] = memories_rewards[nearest_memories].cpu().numpy()

            updated_train_paths.append(new_path)
        print(f'Updating train paths took {time.time() - t0} seconds')

        # save
        data = {'train_paths': updated_train_paths, 
            'memories_obs': memories.cpu().numpy(), 
            'memories_act': memories_actions.cpu().numpy(), 
            'memories_next_obs': memories_next_obs.cpu().numpy(),
            'memories_rewards': memories_rewards.cpu().numpy()}
        with open(save_name, 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f'{save_name} already exists. Skipping.')

