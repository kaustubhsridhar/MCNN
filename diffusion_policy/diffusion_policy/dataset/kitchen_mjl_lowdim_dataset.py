from typing import Dict
import torch
import numpy as np
import copy, os, pickle
import pathlib
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs

class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0,
            num_memories_frac=0.1,
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        if not os.path.exists(f'{dataset_dir}/replay_buffer.pkl'):
            robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
                0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
                0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
                0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
            rng = np.random.default_rng(seed=seed)

            data_directory = pathlib.Path(dataset_dir)
            self.replay_buffer = ReplayBuffer.create_empty_numpy()
            for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
                try:
                    data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                    qpos = data['qpos'].astype(np.float32)
                    obs = np.concatenate([
                        qpos[:,:9],
                        qpos[:,-21:],
                        np.zeros((len(qpos),30),dtype=np.float32)
                    ], axis=-1)
                    if robot_noise_ratio > 0:
                        # add observation noise to match real robot
                        noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                            low=-1., high=1., size=(obs.shape[0], 30))
                        obs[:,:30] += noise
                    episode = {
                        'obs': obs,
                        'action': data['ctrl'].astype(np.float32)
                    }
                    self.replay_buffer.add_episode(episode)
                except Exception as e:
                    print(i, e)

            with open(f'{dataset_dir}/replay_buffer.pkl', 'wb') as f:
                pickle.dump(self.replay_buffer, f)

            print(f"{self.replay_buffer['obs'].shape=}, {self.replay_buffer['action'].shape=}")
            np.save(f'{dataset_dir}/all_observations_multitask.npy', self.replay_buffer['obs'])
            np.save(f'{dataset_dir}/all_actions_multitask.npy', self.replay_buffer['action'])
        else:
            with open(f'{dataset_dir}/replay_buffer.pkl', 'rb') as f:
                self.replay_buffer_og = pickle.load(f)

            all_observations = np.load(f'{dataset_dir}/all_observations_multitask.npy')
            all_actions = np.load(f'{dataset_dir}/all_actions_multitask.npy')

            loc = f'../mems_obs/updated_datasets/kitchen/updated_{num_memories_frac}_frac.pkl'
            if os.path.exists(loc):
                with open(loc, 'rb') as f:
                    updated_data = pickle.load(f)
                mem_observations = updated_data['mem_observations'] # memory and memory_action for every datapoint
                mem_actions = updated_data['mem_actions']
            else:
                print(f"Skipping loading memories since they don't (yet) exist at {loc}.")
            
            self.replay_buffer = ReplayBuffer.create_empty_numpy()
            start_idx = 0
            for end_idx in self.replay_buffer_og.episode_ends:
                episode = {
                    'obs': all_observations[start_idx:end_idx],
                    'action': all_actions[start_idx:end_idx],
                    'mem_observations': mem_observations[start_idx:end_idx],
                    'mem_actions': mem_actions[start_idx:end_idx]
                }
                self.replay_buffer.add_episode(episode)
                start_idx = end_idx

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action'],
            'mem_observations': self.replay_buffer['obs'], # so that we use the same normalizer for both obs and mem_obs
            'mem_actions': self.replay_buffer['action'], # so that we use the same normalizer for both act and mem_act
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
