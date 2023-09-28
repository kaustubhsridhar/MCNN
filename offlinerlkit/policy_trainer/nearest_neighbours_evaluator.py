import time
import os

import numpy as np
import torch
import torch.nn as nn
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from copy import deepcopy

class NearestNeighboursEvaluator:
    def __init__(
        self,
        algo: str,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        batch_size: int = 5000,
        eval_episodes: int = 10,
        device: str = "cpu",
        perception_model: nn.Module = None,
        use_tqdm: bool = True,
    ) -> None:
        self.algo = algo
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.device = device

        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.perception_model = perception_model
        self.use_tqdm = use_tqdm
        self.k = 10 # num of closest neighbours to consider for VINN

    def one_nearest_neighbours(self, obs: np.ndarray):
        if obs.ndim == 1:
            obs = obs[None, :]
        assert obs.shape[0] == 1
        
        train_data = self.buffer.sample_all()
        train_obss, train_actions, train_rewards = train_data['observations'], train_data['actions'], train_data['rewards']
        assert train_obss.ndim == obs.ndim == 2

        # find closest observation
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        train_obss = torch.as_tensor(train_obss, device=self.device, dtype=torch.float32)
        least_dist = np.inf
        for start in range(0, len(train_obss), self._batch_size):
            end = min(start+self._batch_size, len(train_obss))
            batch = {'obss': train_obss[start:end], 'actions': train_actions[start:end], 'rewards': train_rewards[start:end]}
            dist, closest_obs_idx = torch.cdist(obs, batch['obss']).min(dim=1)
            if dist.item() < least_dist:
                least_dist = dist.item()
                closest_action = deepcopy(batch['actions'][closest_obs_idx])

        return closest_action
    
    def VINN(self, obs: np.ndarray):
        if obs.ndim == 1:
            obs = obs[None, :]
        assert obs.shape[0] == 1
        
        train_data = self.buffer.sample_all()
        train_obss, train_actions, train_rewards = train_data['observations'], train_data['actions'], train_data['rewards']
        assert train_obss.ndim == obs.ndim == 2

        # find closest observation
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        train_obss = torch.as_tensor(train_obss, device=self.device, dtype=torch.float32)
        all_closest_k_indices_and_dists = []
        for start in range(0, len(train_obss), self._batch_size):
            end = min(start+self._batch_size, len(train_obss))
            batch = {'obss': train_obss[start:end], 'actions': train_actions[start:end], 'rewards': train_rewards[start:end]}
            sorted_dists, closest_indices = torch.cdist(obs, batch['obss']).sort(dim=1) # torch.Size([1, batch_size]) torch.Size([1, batch_size])
            
            closest_k_indices = closest_indices[0, :self.k].cpu().numpy() + start
            closest_k_dists = sorted_dists[0, :self.k].cpu().numpy()
            all_closest_k_indices_and_dists.extend(list(zip(closest_k_indices, closest_k_dists)))

        # find top k from all closest k indices
        top_k_indices = sorted(all_closest_k_indices_and_dists, key=lambda x: x[1])[:self.k]

        # find VINN action as weighted average of top k actions
        top_k_actions = np.array([np.exp(-d) * train_actions[i] for i, d in top_k_indices])
        sum_of_distances = np.sum([np.exp(-d) for _, d in top_k_indices])
        closest_action = np.sum(top_k_actions, axis=0) / sum_of_distances

        return closest_action

    def evaluate(self):
        start_time = time.time()
        
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            with torch.no_grad():
                if self.perception_model is not None:
                    obs = self.perception_model(obs)
            if self.algo == "1nn":
                action = self.one_nearest_neighbours(obs) 
            elif self.algo == "vinn":
                action = self.VINN(obs)
            else:
                raise NotImplementedError
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        eval_info =  {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
        norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
        self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
        self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
        self.logger.logkv("eval/episode_length", ep_length_mean)
        self.logger.logkv("eval/episode_length_std", ep_length_std)
        self.logger.set_timestep(0)
        self.logger.dumpkvs()

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.logger.close()

