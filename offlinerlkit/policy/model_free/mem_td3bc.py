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
from copy import deepcopy

class MemTD3BCPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: MemActor,
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
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

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
        with torch.no_grad():
            dist = self.compute_distances(obs, mem_obs)
            action = self.actor(obs, mem_actions, dist, beta=0).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def compute_distances(self, obs: np.ndarray, mem_obs: np.ndarray, dim: int = 1, pnorm: int = 2) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        mem_obs = torch.as_tensor(mem_obs, device=self.device, dtype=torch.float32)
        dist = torch.norm(obs - mem_obs, p=pnorm, dim=dim).unsqueeze(1)
        return dist

    
    def learn(self, batch: Dict) -> Dict[str, float]:
        self.iter += 1

        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        mem_observations = batch["mem_observations"]
        mem_actions = batch["mem_actions"]
        # mem_next_observations = batch["mem_next_observations"]
        with torch.no_grad():
            dist = self.compute_distances(obss, mem_observations)
        
        beta = 0
        if self.len_of_beta_step != 0 and self.num_beta_steps != 0:
            beta = max(0.99 * (1 - int(self.iter / self.len_of_beta_step)/self.num_beta_steps), 0)

        if self.only_bc:
            # update only actor
            a = self.actor(obss, mem_actions, dist, beta)
            actor_loss = ((a - actions).pow(2)).mean()
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
            # prepare next 
            with torch.no_grad():
                next_obss_temp = deepcopy(next_obss)
                if self.scaler is not None:
                    next_obss_temp = self.scaler.inverse_transform(next_obss_temp.cpu().numpy())
                next_mem_observations, next_mem_actions = self.find_memories(next_obss_temp)
                if self.scaler is not None:
                    next_mem_observations = self.scaler.transform(next_mem_observations)
                next_dist = self.compute_distances(next_obss, next_mem_observations)

            # update critic
            q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
            with torch.no_grad():
                noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
                next_actions = (self.actor_old(next_obss, next_mem_actions, next_dist, beta) + noise).clamp(-self._max_action, self._max_action)
                next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
                target_q = rewards + self._gamma * (1 - terminals) * next_q
            
            critic1_loss = ((q1 - target_q).pow(2)).mean()
            critic2_loss = ((q2 - target_q).pow(2)).mean()

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()

            # update actor
            if self._cnt % self._freq == 0:
                a = self.actor(obss, mem_actions, dist, beta)
                q = self.critic1(obss, a)
                lmbda = self._alpha / q.abs().mean().detach()
                actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                self._last_actor_loss = actor_loss.item()
                self._sync_weight()
        
            self._cnt += 1

            return {
                "loss/actor": self._last_actor_loss,
                "loss/critic1": critic1_loss.item(),
                "loss/critic2": critic2_loss.item()
            }