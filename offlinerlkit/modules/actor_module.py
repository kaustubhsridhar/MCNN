import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions
    

def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)
    
class MemActor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        Lipz: float = 1.0,
        lamda: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action
        self.Lipz = Lipz
        self.lamda = lamda
        self.memory_act = self.shifted_crazy_relu

    def shifted_sigmoid(self, x, beta):
        return 2 * nn.Sigmoid()(x) - 1

    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1

    def forward(self, obs, mem_actions, dist, beta) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        mem_actions = torch.as_tensor(mem_actions, device=self.device, dtype=torch.float32)
        dist = torch.as_tensor(dist, device=self.device, dtype=torch.float32)
        
        logits = self.backbone(obs)
        # outputs = self._max * torch.tanh(self.last(logits))

        lamda_in_exp = self.lamda * 10 
        exp_lamda_dist = torch.exp(-lamda_in_exp * dist)
        actions = mem_actions * exp_lamda_dist + self.Lipz * (1-exp_lamda_dist) * self._max * self.memory_act(self.last(logits), beta)
        return actions