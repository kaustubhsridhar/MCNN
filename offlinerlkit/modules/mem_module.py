import numpy as np
import torch
import torch.nn as nn
from offlinerlkit.nets import MLP
from typing import Dict, List, Union, Tuple, Optional

def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)

class MemDynamicsModel(MLP):

    def __init__(self, 
                input_dim: int,
                hidden_dims: Union[List[int], Tuple[int]],
                output_dim: Optional[int] = None,
                activation: nn.Module = nn.ReLU,
                dropout_rate: Optional[float] = None,
                Lipz: float = 100.0,
                lamda: float = 1.0,
                device: str = "cuda",
    ):
        super().__init__(input_dim, hidden_dims, output_dim, activation, dropout_rate)
        self.memory_act = self.shifted_crazy_relu
        self.Lipz = Lipz
        self.lamda = lamda
        self.device = torch.device(device)

        self.to(self.device)

    def shifted_sigmoid(self, x, beta):
        return 2 * nn.Sigmoid()(x) - 1

    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1

    def forward(self, inputs, mem_targets, dist, beta):
        lamda_in_exp = self.lamda * 10 # self.lamda * self.Lipz * 10
        exp_lamda_dist = torch.exp(-lamda_in_exp * dist)
        outputs = self.model(inputs)
        preds = mem_targets * exp_lamda_dist + self.Lipz * (1-exp_lamda_dist) * self.memory_act(
                                                                    outputs,
                                                                    beta,
                                                                )
        
        return preds


