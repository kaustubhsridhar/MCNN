"""
Credits: https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md
"""

import gym
import numpy as np

import collections
import pickle

import d4rl
import os
import torch 
from copy import deepcopy
from tqdm import tqdm

from offlinerlkit.carla.carla_model import CoILICRA
from offlinerlkit.carla.carla_config import MODEL_CONFIGURATION

def generate_embeddings(name):
    with open(f'data/datasets/{name}.pkl', 'rb') as f:
        paths = pickle.load(f)

    # load model from data/models/nocrash/resnet34imnet10S1/checkpoints/660000.pth
    model_data = torch.load('data/models/nocrash/resnet34imnet10S1/checkpoints/660000.pth')
    model_state_dict = model_data['state_dict']
    model = CoILICRA(MODEL_CONFIGURATION)
    model.load_state_dict(model_state_dict)
    model.eval()

    new_paths = []
    for path in tqdm(paths):
        new_path = deepcopy(path)
        
        obs = path['observations']
        next_obs = path['next_observations']
        
        batch_size = obs.shape[0]
        
        obs = obs.reshape(batch_size, 48, 48, 3)
        next_obs = next_obs.reshape(batch_size, 48, 48, 3)

        # print(f'{obs.max()=}, {obs.min()=}')
        # print(f'{next_obs.max()=}, {next_obs.min()=}')
        
        inputs = torch.tensor(obs).permute(0, 3, 1, 2).float()
        # print(f'{inputs.max()=}, {inputs.min()=}')
        embeddings, _ = model.perception(inputs)
        new_path['embeddings'] = embeddings.detach().numpy()
        
        next_inputs = torch.tensor(next_obs).permute(0, 3, 1, 2).float()
        # print(f'{next_inputs.max()=}, {next_inputs.min()=}')
        next_embeddings, _ = model.perception(next_inputs)
        new_path['next_embeddings'] = next_embeddings.detach().numpy()
        
        new_paths.append(new_path)

    with open(f'data/datasets/{name}_embeddings.pkl', 'wb') as f:
        pickle.dump(new_paths, f)

    print(f'Generated embeddings for {name} dataset')

if __name__ == '__main__':
    for name in ['carla-lane-v0', 'carla-town-v0']:
        generate_embeddings(name)