# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb

def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)

class MCNN_MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16,
                 Lipz=1.0,
                 lamda=1.0,
                 max_action=1.0):

        super(MCNN_MLP, self).__init__()
        self.device = device
        self.Lipz = Lipz
        self.lamda = lamda

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)
        self.beta = 0
        self._max = max_action
        self.memory_act = self.shifted_crazy_relu

    def shifted_sigmoid(self, x, beta):
        return 2 * nn.Sigmoid()(x) - 1

    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1

    def backbone(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        out = self.final_layer(x) 
        return out

    def forward(self, x, time, state, mem_state, mem_action, dist):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        mem_action = torch.as_tensor(mem_action, device=self.device, dtype=torch.float32)
        dist = torch.as_tensor(dist, device=self.device, dtype=torch.float32)

        logits = self.backbone(x, time, state)

        lamda_in_exp = self.lamda * 10.0
        exp_lamda_dist = torch.exp(- lamda_in_exp * dist)
        out = mem_action * exp_lamda_dist + self.Lipz * (1-exp_lamda_dist) * self._max * self.memory_act(logits, self.beta)

        return out


