import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict
import bisect
import random

class SequenceReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu",
        window_size: int = 1,
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.mem_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.mem_next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.mem_actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.mem_rewards = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)
        self.window_size = window_size

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()
        self.mem_observations[self._ptr] = np.array(obs).copy()
        self.mem_next_observations[self._ptr] = np.array(next_obs).copy()
        self.mem_actions[self._ptr] = np.array(action).copy()
        self.mem_rewards[self._ptr] = np.array(reward).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()
        self.mem_observations[indexes] = np.array(obss).copy()
        self.mem_next_observations[indexes] = np.array(next_obss).copy()
        self.mem_actions[indexes] = np.array(actions).copy()
        self.mem_rewards[indexes] = np.array(rewards).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        mem_observations = np.array(dataset["mem_observations"], dtype=self.obs_dtype)
        mem_next_observations = np.array(dataset["mem_next_observations"], dtype=self.obs_dtype)
        mem_actions = np.array(dataset["mem_actions"], dtype=self.action_dtype)
        mem_rewards = np.array(dataset["mem_rewards"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.mem_observations = mem_observations
        self.mem_next_observations = mem_next_observations
        self.mem_actions = mem_actions
        self.mem_rewards = mem_rewards

        self._ptr = len(observations)
        self._size = len(observations)

        self.zero_and_terminal_idxs = [0] + list(np.where(self.terminals==1)[0]) + ([self._size] if self.terminals[-1] != 1 else [])
        self.traj_start_ends = [(self.zero_and_terminal_idxs[i], self.zero_and_terminal_idxs[i+1]) for i in range(len(self.zero_and_terminal_idxs)-1)]
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        self.mem_observations = (self.mem_observations - mean) / std
        self.mem_next_observations = (self.mem_next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        # Updated sampling method
        sampled_trajectories = random.choices(self.traj_start_ends, k=batch_size) 
        batch_start_indexes = [np.random.randint(start, end-self.window_size) for (start, end) in sampled_trajectories] # low (included), high (excluded)
        seq_start_end = [(idx, idx+self.window_size) for idx in batch_start_indexes]
        obs_seq_batch = np.array([self.observations[start:end] for (start, end) in seq_start_end])
        action_seq_batch = np.array([self.actions[start:end] for (start, end) in seq_start_end])
        mem_obs_seq_batch = np.array([self.mem_observations[start:end] for (start, end) in seq_start_end])
        mem_action_seq_batch = np.array([self.mem_actions[start:end] for (start, end) in seq_start_end])
        
        return {
            "observations_seq": torch.tensor(obs_seq_batch).to(self.device),
            "actions_seq": torch.tensor(action_seq_batch).to(self.device),
            "mem_observations_seq": torch.tensor(mem_obs_seq_batch).to(self.device),
            "mem_actions_seq": torch.tensor(mem_action_seq_batch).to(self.device),
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),
            "mem_observations": self.mem_observations[:self._size].copy(),
            "mem_actions": self.mem_actions[:self._size].copy(),
            "mem_next_observations": self.mem_next_observations[:self._size].copy(),
            "mem_rewards": self.mem_rewards[:self._size].copy(),
        }