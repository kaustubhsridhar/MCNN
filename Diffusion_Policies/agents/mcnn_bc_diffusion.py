# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from agents.mcnn_diffusion import MCNN_Diffusion
from agents.mcnn_model import MCNN_MLP


class Diffusion_MCNN_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=2e-4,
                 Lipz=1.0,
                 lamda=1.0,
                 scaler=None,
                 dataset=None
                 ):

        self.model = MCNN_MLP(state_dim=state_dim, action_dim=action_dim, device=device, Lipz=Lipz, lamda=lamda)
        self.actor = MCNN_Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, predict_epsilon=False # so that loss is between action and prediction (and not action and epsilon)
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.Lipz = Lipz
        self.lamda = lamda
        self.scaler = scaler

        self.nodes_states = torch.as_tensor(dataset["memories_obs"], device=self.device, dtype=torch.float32)
        self.nodes_actions = torch.as_tensor(dataset["memories_actions"], device=self.device, dtype=torch.float32)
        assert self.nodes_states.ndim == self.nodes_actions.ndim == 2

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            batch = replay_buffer.sample(batch_size)

            state, action, next_state, reward, dones = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
            not_done = 1 - dones
        
            mem_state = batch["mem_observations"]
            mem_action = batch["mem_actions"]

            with torch.no_grad():
                dist = self.compute_distances(state, mem_state)

            loss = self.actor.loss(x=action, state=state, mem_state=mem_state, mem_action=mem_action, dist=dist)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric
    
    def find_memories(self, state):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        _, closest_nodes = torch.cdist(state, self.nodes_states).min(dim=1)
        mem_state = self.nodes_states[closest_nodes, :]
        mem_action = self.nodes_actions[closest_nodes, :]

        return mem_state.cpu().numpy(), mem_action

    def compute_distances(self, state, mem_state):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        mem_state = torch.as_tensor(mem_state, device=self.device, dtype=torch.float32)
        dist = torch.norm(state - mem_state, p=2, dim=1).unsqueeze(1)
        return dist

    def sample_action(self, state):
        state = state.reshape(1, -1)
        
        ##### 
        with torch.no_grad():
            mem_state, mem_action = self.find_memories(state)
            if self.scaler is not None:
                mem_state = self.scaler.transform(mem_state)
        if self.scaler is not None:
            state = self.scaler.transform(state)
        with torch.no_grad():
            dist = self.compute_distances(state, mem_state)
    
            #####
            action = self.actor.sample(state, mem_state=mem_state, mem_action=mem_action, dist=dist)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))

