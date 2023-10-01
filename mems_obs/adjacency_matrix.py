
import pickle
import numpy as np
import torch
from collections import defaultdict
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc

num_memories_frac = 0.1
task = f'pen-human-v1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

location_of_gng_file = f'mems_obs/memories/{task}-1.0/memories_{num_memories_frac}_frac.pkl'

# load gng nodes 
with open(f'{location_of_gng_file}', 'rb') as f:
    gng = pickle.load(f)
nodes = {}
for i, n in enumerate(gng.graph.edges_per_node.keys()):
    nodes[n] = (i, n.weight)

# load gng edges and create adjacency matrix for gng 
adjacency_matrix = np.zeros((len(nodes), len(nodes)))
for i, (n, other_nodes) in enumerate(gng.graph.edges_per_node.items()):
    for other_n in other_nodes:
        j = nodes[other_n][0]
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
print(f'{adjacency_matrix.shape=}') 

# load updated datasets' memories
dataset = qlearning_dataset_percentbc(task, 1.0, num_memories_frac)
observations = dataset['observations']
saved_memories_array = dataset['memories_obs']

# checks
assert len(nodes) == len(saved_memories_array)

# Why?
# Each memory (ie row of saved_memories_array) is the observation (ie some row of observations) that was closest to the node (ie row of nodes)
# The adjacency matrix is the connection in the gng and remains the same for both the gng and the memories

