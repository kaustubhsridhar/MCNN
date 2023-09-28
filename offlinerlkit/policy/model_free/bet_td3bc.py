import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from miniBET.behavior_transformer import BehaviorTransformer
from collections import deque

class BetTD3BCPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: BehaviorTransformer,
        critic1: nn.Module,
        critic2: nn.Module,
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

        self.device = self.actor.device

        self.only_bc = only_bc
        self._alpha = alpha
        assert (self.only_bc and self._alpha == 0.0) or (not self.only_bc and self._alpha > 0.0)
        self.scaler = scaler
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

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            if self.perception_model is not None:
                obs = self.perception_model(obs)
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        
        self.obs_buffer.append(obs)
        obs_seq = torch.as_tensor(np.stack(self.obs_buffer), dtype=torch.float32).to(self.device) # (1, T, obs_dim)
        
        with torch.no_grad():
            eval_action_seq, eval_loss, eval_loss_dict = self.actor(obs_seq=obs_seq, action_seq=None, goal_seq=None)
            action = eval_action_seq[0, -1, :].cpu().numpy() # (1, T, action_dim) --> (1, action_dim)
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:

        obs_seq, action_seq = batch["observations_seq"], batch["actions_seq"]
        
        if self.only_bc:
            # update only actor

            # OLD:
            # a = self.actor(obss)
            # actor_loss = ((a - actions).pow(2)).mean()

            # NEW:
            train_action, actor_loss, actor_loss_dict = self.actor(obs_seq=obs_seq, action_seq=action_seq, goal_seq=None)
            # print(train_action.shape) # (N, T, action_dim)

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