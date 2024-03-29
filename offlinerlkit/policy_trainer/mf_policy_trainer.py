import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
import time
import skvideo.io

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        sequential: bool = False,
        save_videos: bool = False,
        freq_save_videos: int = 50,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.sequential = sequential
        self.save_videos = save_videos
        self.freq_save_videos = freq_save_videos
        self.videos_dir = os.path.join(self.logger._dir, "saved_videos")
        if self.save_videos:
            os.makedirs(self.videos_dir, exist_ok=True)

    def train(self, use_tqdm: bool = True) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            # train
            t0 = time.time()
            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}") if use_tqdm else range(self._step_per_epoch)
            for it in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                if use_tqdm: pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.logkv("train/time", time.time() - t0)
            
            # evaluate current policy
            t1 = time.time()
            eval_info = self._evaluate(epoch=e)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.logkv("eval/time", time.time() - t1)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self, epoch=None) -> Dict[str, List[float]]:
        self.policy.eval()
        if self.sequential: self.policy.reset_obs_buffer()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        
        save_videos_this_epoch = self.save_videos and (epoch is not None) and ((epoch+1) % self.freq_save_videos == 0)
        self.logger.logkv("eval/save_videos_this_epoch", save_videos_this_epoch)
        if save_videos_this_epoch:
            arrs = []

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())

            if save_videos_this_epoch:
                curr_frame = self.eval_env.sim.render(width=640, height=480, mode='offscreen', camera_name=None, device_id=0)
                arrs.append(curr_frame[::-1, :, :])

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if self.sequential: self.policy.reset_obs_buffer()
                obs = self.eval_env.reset()
                if save_videos_this_epoch:
                    skvideo.io.vwrite( f'{self.videos_dir}/epoch_{epoch+1}_episode_{num_episodes+1}.mp4', np.asarray(arrs))
                    arrs = []
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def get_all_reward_and_dists_from_memories(self, memories_obs: torch.Tensor):
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        all_rewards = []
        all_dists = []

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            with torch.no_grad():
                if self.policy.perception_model is not None:
                    obs = self.policy.perception_model(obs)
            dist_to_closest_memory, _ = torch.cdist(torch.as_tensor(obs.reshape(1,-1), device=self.policy.actor.device, dtype=torch.float32), memories_obs, p=2).min(dim=1)
            all_rewards.append(reward)
            all_dists.append(dist_to_closest_memory.item())

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        eval_info = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
        
        return all_rewards, all_dists, norm_ep_rew_mean
