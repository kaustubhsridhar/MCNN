import numpy as np
import torch
import os 
from neural_gas_helpers import data2gas
import pickle
from tqdm import tqdm
from copy import deepcopy
import time

def create_gng_incremental_for_frankakitchen(folder, args):
    all_observations = np.load(f'diffusion_policy/data/{args.name}/kitchen_demos_multitask/all_observations_multitask.npy')
    print(f'{all_observations.shape=}')
    gng = data2gas(states=all_observations, max_memories=int(len(all_observations) * args.num_memories_frac), gng_epochs=args.gng_epochs)
    os.makedirs(f'{folder}/{args.name}', exist_ok=True)
    with open(f'{folder}/{args.name}/memories_{args.num_memories_frac}_frac.pkl', 'wb') as f:
        pickle.dump(gng, f)
    print(f'Done.')

def update_data_for_frankakitchen(folder, args):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load all observations and actions
    all_observations = np.load(f'diffusion_policy/data/{args.name}/kitchen_demos_multitask/all_observations_multitask.npy')
    all_observations_tensor = deepcopy(all_observations)
    all_observations_tensor = torch.from_numpy(all_observations_tensor).float().to(device)
    all_actions = np.load(f'diffusion_policy/data/{args.name}/kitchen_demos_multitask/all_actions_multitask.npy')
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