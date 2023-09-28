import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from copy import deepcopy 


class AWRPolicy(TD3Policy):
    """
    AWR <Ref: https://github.com/jcwleo/awr-pytorch>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        scaler: StandardScaler = None,
        gamma = 0.99,
        lam = 0.95,
        beta = 0.05,
        max_weight = 20.0,
        use_gae = True,
    ) -> None:

        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_old = deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_old = deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self.scaler = scaler

        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.max_weight = max_weight
        self.use_gae = use_gae
    
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
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def learn_critic(self, batch: Dict) -> Dict[str, float]:
        s_batch, action_batch, next_s_batch, reward_batch, done_batch = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        data_len = len(s_batch)
        mse = nn.MSELoss()

        # forward
        cur_value = self.critic1(s_batch)
        print('Before opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, _ = self.discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        # discounted_reward = (discounted_reward - discounted_reward.mean())/(discounted_reward.std() + 1e-8)
        
        # backward
        sample_idx = random.sample(range(data_len), 256)
        sample_value = self.critic1(s_batch[sample_idx])
        if (torch.sum(torch.isnan(sample_value)) > 0):
            print('NaN in value prediction')
            input()
        critic_loss = mse(sample_value.squeeze(), torch.FloatTensor(discounted_reward[sample_idx]))
        self.critic1_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()

    def learn_actor(self, batch: Dict) -> Dict[str, float]:
        s_batch, action_batch, next_s_batch, reward_batch, done_batch = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        data_len = len(s_batch)
        mse = nn.MSELoss()

        # forward
        cur_value = self.critic1(s_batch)
        print('After opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, adv = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        print('Advantage has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(adv).float()))))
        print('Returns has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(discounted_reward).float()))))
        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # backward
        sample_idx = random.sample(range(data_len), 256)
        weight = torch.tensor(np.minimum(np.exp(adv[sample_idx] / beta), max_weight)).float().reshape(-1, 1)
        cur_policy = self.model.actor(torch.FloatTensor(s_batch[sample_idx]))

        probs = -cur_policy.log_probs(torch.tensor(action_batch[sample_idx]).float())
        actor_loss = probs * weight

        actor_loss = actor_loss.mean()
        # print(actor_loss)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        print('Weight has nan {}'.format(torch.sum(torch.isnan(weight))))
