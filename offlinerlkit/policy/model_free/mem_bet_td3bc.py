import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.modules.actor_module import MemActor
from offlinerlkit.modules.critic_module import Critic
from miniBET.behavior_transformer import BehaviorTransformer
from copy import deepcopy
from collections import deque

class MemBetTD3BCPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: BehaviorTransformer,
        critic1: Critic,
        critic2: Critic,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
        scaler: StandardScaler = None,
        only_bc: bool = False,
        dataset: Dict = None,
        perception_model: nn.Module = None,
        window_size: int = 1,
    ) -> None:

        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq
        )

        self.only_bc = only_bc
        self._alpha = alpha
        assert (self.only_bc and self._alpha == 0.0) or (not self.only_bc and self._alpha > 0.0)
        self.scaler = scaler
        self.iter = 0
        self.len_of_beta_step = 10
        self.num_beta_steps = 10

        self.device = self.actor.device

        self.nodes_obs = torch.as_tensor(dataset["memories_obs"], device=self.device, dtype=torch.float32)
        self.nodes_actions = torch.as_tensor(dataset["memories_actions"], device=self.device, dtype=torch.float32)
        self.nodes_next_obs = torch.as_tensor(dataset["memories_next_obs"], device=self.device, dtype=torch.float32)
        self.nodes_rewards = torch.as_tensor(dataset["memories_rewards"], device=self.device, dtype=torch.float32).unsqueeze(1)
        assert self.nodes_obs.ndim == self.nodes_actions.ndim == self.nodes_next_obs.ndim == self.nodes_rewards.ndim == 2
        self.perception_model = perception_model
        self.window_size = window_size
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def reset_obs_buffer(self):
        self.obs_buffer = deque(maxlen=self.window_size)
        self.mem_obs_buffer = deque(maxlen=self.window_size)
        self.mem_action_buffer = deque(maxlen=self.window_size)

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def find_memories(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        _, closest_nodes = torch.cdist(obs, self.nodes_obs).min(dim=1)
        mem_obs = self.nodes_obs[closest_nodes, :]
        mem_actions = self.nodes_actions[closest_nodes, :]
        mem_next_obss = self.nodes_next_obs[closest_nodes, :]
        mem_rewards = self.nodes_rewards[closest_nodes, :]

        return mem_obs.cpu().numpy(), mem_actions
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:        
        with torch.no_grad():
            if self.perception_model is not None:
                obs = self.perception_model(obs)
            mem_obs, mem_actions = self.find_memories(obs)
            if self.scaler is not None:
                mem_obs = self.scaler.transform(mem_obs)
        if self.scaler is not None:
            obs = self.scaler.transform(obs)

        self.obs_buffer.append(obs)
        self.mem_obs_buffer.append(mem_obs)
        self.mem_action_buffer.append(mem_actions.cpu().numpy())

        obs_seq = torch.as_tensor(np.stack(self.obs_buffer), dtype=torch.float32).to(self.device) # (1, T, obs_dim)
        mem_obs_seq = torch.as_tensor(np.stack(self.mem_obs_buffer), dtype=torch.float32).to(self.device) # (1, T, obs_dim)
        mem_action_seq = torch.as_tensor(np.stack(self.mem_action_buffer), dtype=torch.float32).to(self.device) # (1, T, obs_dim)

        with torch.no_grad():
            dist_seq = self.compute_distances(obs_seq, mem_obs_seq, dim=2)
            eval_action_seq, eval_loss, eval_loss_dict = self.actor(obs_seq=obs_seq, action_seq=None, goal_seq=None, mem_action_seq=mem_action_seq, dist_seq=dist_seq, beta=0)
            action = eval_action_seq[0, -1, :].cpu().numpy() # (1, T, action_dim) --> (1, action_dim)
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def compute_distances(self, obs: np.ndarray, mem_obs: np.ndarray, dim: int = 1, pnorm: int = 2) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        mem_obs = torch.as_tensor(mem_obs, device=self.device, dtype=torch.float32)
        dist = torch.norm(obs - mem_obs, p=pnorm, dim=dim).unsqueeze(dim)
        return dist
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        self.iter += 1
        
        # mem_observations, mem_actions = batch["mem_observations"], batch["mem_actions"]
        obs_seq, action_seq = batch["observations_seq"], batch["actions_seq"]
        mem_obs_seq, mem_action_seq = batch["mem_observations_seq"], batch["mem_actions_seq"]

        with torch.no_grad():
            # dist = self.compute_distances(obss, mem_observations)
            dist_seq = self.compute_distances(obs_seq, mem_obs_seq, dim=2)
        
        beta = 0
        if self.len_of_beta_step != 0 and self.num_beta_steps != 0:
            beta = max(0.99 * (1 - int(self.iter / self.len_of_beta_step)/self.num_beta_steps), 0)

        if self.only_bc:
            # update only actor

            # OLD:
            # a = self.actor(obss)
            # actor_loss = ((a - actions).pow(2)).mean()

            # NEW:
            train_action, actor_loss, actor_loss_dict = self.actor(obs_seq=obs_seq, 
                                                                   action_seq=action_seq, 
                                                                   goal_seq=None,
                                                                   mem_action_seq=mem_action_seq,
                                                                   dist_seq=dist_seq,
                                                                   beta=beta,
                                                                )

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()

            return {
                "loss/actor": self._last_actor_loss,
                "loss/critic1": -1,
                "loss/critic2": -1
            }
        else:
            raise NotImplementedError