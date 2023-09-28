import numpy as np
import torch
import collections
import pickle 

def qlearning_dataset_percentbc(task, chosen_percentage, num_memories_frac, use_random_memories=False, prefix=None):
    if chosen_percentage < 1.0:
        return qlearning_dataset_smaller_percentbc(task, chosen_percentage, num_memories_frac, use_random_memories=use_random_memories)
    
    full_name = f'mems_obs/updated_datasets/{task}-1.0/{"random_" if use_random_memories else ""}updated_{num_memories_frac}_frac.pkl'
    if prefix is not None: # only for Diffusion_Policies
         full_name = f'{prefix}{full_name}'
    with open(full_name, 'rb') as f:
            data = pickle.load(f)

    train_paths, memories_obs, memories_act, memories_next_obs, memories_rewards = data['train_paths'], data['memories_obs'], data['memories_act'], data['memories_next_obs'], data['memories_rewards']

    choice = 'embeddings' if 'carla' in task else 'observations'

    obs_dim = train_paths[0][choice].shape[1]
    act_dim = train_paths[0]['actions'].shape[1]
    print(f'{obs_dim=}, {act_dim=}')

    observations = np.concatenate([path[choice] for path in train_paths], axis=0)
    actions = np.concatenate([path['actions'] for path in train_paths], axis=0)
    next_observations = np.concatenate([path[f'next_{choice}'] for path in train_paths], axis=0)
    rewards = np.concatenate([path['rewards'] for path in train_paths], axis=0)
    
    for path in train_paths:
        path['terminals'][-1] = 1.0
    terminals = np.concatenate([path['terminals'] for path in train_paths], axis=0)

    mem_observations = np.concatenate([path['mem_observations'] for path in train_paths], axis=0)
    mem_actions = np.concatenate([path['mem_actions'] for path in train_paths], axis=0)
    mem_next_observations = np.concatenate([path['mem_next_observations'] for path in train_paths], axis=0)
    mem_rewards = np.concatenate([path['mem_rewards'] for path in train_paths], axis=0)

    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'rewards': rewards,
        'terminals': terminals,
        'mem_observations': mem_observations,
        'mem_actions': mem_actions,
        'mem_next_observations': mem_next_observations,
        'mem_rewards': mem_rewards,
        'memories_obs': memories_obs,
        'memories_actions': memories_act,
        'memories_next_obs': memories_next_obs,
        'memories_rewards': memories_rewards,
    }

def qlearning_dataset_smaller_percentbc(task, chosen_percentage, num_memories_frac, use_random_memories=False):
    
    full_name = f'mems_obs/updated_datasets/{task}-1.0/{"random_" if use_random_memories else ""}updated_{num_memories_frac}_frac.pkl'
    with open(full_name, 'rb') as f:
            data = pickle.load(f)

    train_paths, memories_obs, memories_act, memories_next_obs, memories_rewards = data['train_paths'], data['memories_obs'], data['memories_act'], data['memories_next_obs'], data['memories_rewards']

    choice = 'embeddings' if 'carla' in task else 'observations'

    obs_dim = train_paths[0][choice].shape[1]
    act_dim = train_paths[0]['actions'].shape[1]
    print(f'{obs_dim=}, {act_dim=}')

    observations = np.concatenate([path[choice] for path in train_paths], axis=0)
    actions = np.concatenate([path['actions'] for path in train_paths], axis=0)
    next_observations = np.concatenate([path[f'next_{choice}'] for path in train_paths], axis=0)
    rewards = np.concatenate([path['rewards'] for path in train_paths], axis=0)
    
    for path in train_paths:
        path['terminals'][-1] = 1.0
    terminals = np.concatenate([path['terminals'] for path in train_paths], axis=0)

    mem_observations = np.concatenate([path['mem_observations'] for path in train_paths], axis=0)
    mem_actions = np.concatenate([path['mem_actions'] for path in train_paths], axis=0)
    mem_next_observations = np.concatenate([path['mem_next_observations'] for path in train_paths], axis=0)
    mem_rewards = np.concatenate([path['mem_rewards'] for path in train_paths], axis=0)

    chosen_indices = [i for i in range(len(observations))]
    if chosen_percentage < 1.0:
        indices_of_mem_points = []
        other = []
        for i in range(len(observations)):
            if np.allclose(observations[i], mem_observations[i]):
                indices_of_mem_points.append(i)
            else:
                other.append(i)
        chosen_indices = indices_of_mem_points + other[:int(chosen_percentage * len(observations)) - len(indices_of_mem_points)]

    return {
        'observations': observations[chosen_indices],
        'actions': actions[chosen_indices],
        'next_observations': next_observations[chosen_indices],
        'rewards': rewards[chosen_indices],
        'terminals': terminals[chosen_indices],
        'mem_observations': mem_observations[chosen_indices],
        'mem_actions': mem_actions[chosen_indices],
        'mem_next_observations': mem_next_observations[chosen_indices],
        'mem_rewards': mem_rewards[chosen_indices],
        'memories_obs': memories_obs,
        'memories_actions': memories_act,
        'memories_next_obs': memories_next_obs,
        'memories_rewards': memories_rewards,
    }
