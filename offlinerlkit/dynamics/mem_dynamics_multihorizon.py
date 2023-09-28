import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from torch.utils.data.dataloader import DataLoader
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules import MemDynamicsModel
from copy import deepcopy
import time
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, mem_inputs: np.ndarray, mem_targets: np.ndarray) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.mem_inputs = mem_inputs
        self.mem_targets = mem_targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.inputs[idx], self.targets[idx], self.mem_inputs[idx], self.mem_targets[idx]
    
class MemMultiStepDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, mem_inputs: np.ndarray, mem_targets: np.ndarray, terminals: np.ndarray, horizon: int, inputs_unscaled: np.ndarray) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.mem_inputs = mem_inputs
        self.mem_targets = mem_targets
        self.terminals = terminals
        self.horizon = horizon
        self.inputs_unscaled = inputs_unscaled

        self.obs_dim = targets.shape[1] - 1 # targets also include 1 dim of reward
        self.action_dim = inputs.shape[1] - self.obs_dim

        self.indices = []
        self.ep_pt_map = defaultdict(list)
        ep_idx = 0
        pt_idx = 0
        for terminal in terminals:
            self.indices.append((ep_idx, pt_idx))
            self.ep_pt_map[ep_idx].append(pt_idx)
            pt_idx += 1
            if terminal:
                ep_idx += 1
                pt_idx = 0
        self.num_episodes = ep_idx

    def __len__(self) -> int:
        return len(self.indices) - self.num_episodes * self.horizon

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        list_of_targets = []
        list_of_unscaled_actions = []
        list_of_mem_inputs = []
        list_of_mem_targets = []
        ep_idx, pt_idx = self.indices[idx]
        num_pts_in_ep = len(self.ep_pt_map[ep_idx])
        for i in range(self.horizon):
            list_of_targets.append(self.targets[ep_idx * num_pts_in_ep + pt_idx + i])
            list_of_unscaled_actions.append(self.inputs_unscaled[ep_idx * num_pts_in_ep + pt_idx + i, -self.action_dim:])
            list_of_mem_inputs.append(self.mem_inputs[ep_idx * num_pts_in_ep + pt_idx + i])
            list_of_mem_targets.append(self.mem_targets[ep_idx * num_pts_in_ep + pt_idx + i])
        
        return self.inputs[idx], list_of_unscaled_actions, list_of_targets, self.mem_inputs[idx], self.mem_targets[idx], list_of_mem_inputs, list_of_mem_targets

class MemDynamics(object):
    def __init__(
        self,
        model: MemDynamicsModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        dataset: Dict,
        penalty_coef: float = 0.0,
        train_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim
        self.scheduler = None
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.device = self.model.device
        self.penalty_coef = penalty_coef
        self.train_horizon = train_horizon
        self.len_of_beta_step = 10
        self.num_beta_steps = 10
        self.num_epochs_bw_evals = 1

        self.nodes_obs = torch.from_numpy(dataset["memories_obs"]).float().to(self.device)
        self.nodes_actions = torch.from_numpy(dataset["memories_actions"]).float().to(self.device)
        self.nodes_next_obs = torch.from_numpy(dataset["memories_next_obs"]).float().to(self.device)
        self.nodes_rewards = torch.from_numpy(dataset["memories_rewards"]).float().to(self.device).unsqueeze(1)
        self.nodes_inputs = torch.cat([self.nodes_obs, self.nodes_actions], dim=1)

        self.obss_abs_max = np.max(np.abs(dataset["observations"]), axis=0, keepdims=True)
        self.obss_abs_max_tensor = torch.as_tensor(self.obss_abs_max).to(self.model.device)

        self.diagnostics = {}

    def find_memories(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, closest_nodes = torch.cdist(inputs, self.nodes_inputs).min(dim=1)
        mem_obss = self.nodes_obs[closest_nodes, :]
        mem_actions = self.nodes_actions[closest_nodes, :]
        mem_next_obss = self.nodes_next_obs[closest_nodes, :]
        mem_rewards = self.nodes_rewards[closest_nodes, :]

        delta_mem_obss = (mem_next_obss - mem_obss) / self.obss_abs_max_tensor

        mem_inputs = torch.cat([mem_obss, mem_actions], dim=1)
        mem_targets = torch.cat([delta_mem_obss, mem_rewards], dim=1)
        return mem_inputs, mem_targets

    def step(
        self,
        obss: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        self.model.eval()
        with torch.no_grad():
            inputs = np.concatenate([obss, actions], axis=-1)
            inputs = torch.from_numpy(inputs).float().to(self.device)
            mem_inputs, mem_targets = self.find_memories(inputs)

            inputs = self.scaler.transform_tensor(inputs)
            mem_inputs = self.scaler.transform_tensor(mem_inputs)
            dist = torch.norm(inputs - mem_inputs, dim=1).unsqueeze(1)
            
            preds = self.model(inputs=inputs, mem_targets=mem_targets, dist=dist, beta=0)
            # unnormalize predicted next states
            preds[..., :-1] = preds[..., :-1] * self.obss_abs_max_tensor

            next_obss = preds[:, :-1].cpu().numpy() + obss
            rewards = preds[:, -1:].cpu().numpy()
            terminals = self.terminal_fn(obss, actions, next_obss)
            info = {}
            info["raw_reward"] = rewards
            # dist = dist.cpu().numpy()

            if self.penalty_coef:
                penalty = -1.0 / dist
                # print(f'{rewards.min()=} {rewards.max()=}')
                # print(f'before clip penalty.min()={penalty.min().item()} penalty.max()={penalty.max()=}')
                penalty = np.clip(penalty.cpu().numpy(), -10, 0)
                # print(f'after clip {penalty.min()=} {penalty.max()=} \n')

                assert penalty.shape == rewards.shape
                rewards = rewards - self.penalty_coef * penalty
                info["penalty"] = penalty

            return next_obss, rewards, terminals, info
    
    def format_samples_for_training(self, data: Dict, full_dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        terminals = data["terminals"]
        mem_obss = data["mem_observations"]
        mem_actions = data["mem_actions"]
        mem_next_obss = data["mem_next_observations"]
        mem_rewards = data["mem_rewards"]

        test_obss = full_dataset["test_observations"]
        test_actions = full_dataset["test_actions"]
        test_next_obss = full_dataset["test_next_observations"]
        test_rewards = full_dataset["test_rewards"].reshape(-1, 1)
        test_mem_obss = full_dataset["test_mem_observations"]
        test_mem_actions = full_dataset["test_mem_actions"]
        test_mem_next_obss = full_dataset["test_mem_next_observations"]
        test_mem_rewards = full_dataset["test_mem_rewards"].reshape(-1, 1)

        print(f'obss: {obss.shape}, actions: {actions.shape}, next_obss: {next_obss.shape}, rewards: {rewards.shape}, mem_obss: {mem_obss.shape}, mem_actions: {mem_actions.shape}, mem_next_obss: {mem_next_obss.shape}, mem_rewards: {mem_rewards.shape}')
        print(f'test_obss: {test_obss.shape}, test_actions: {test_actions.shape}, test_next_obss: {test_next_obss.shape}, test_rewards: {test_rewards.shape}, test_mem_obss: {test_mem_obss.shape}, test_mem_actions: {test_mem_actions.shape}, test_mem_next_obss: {test_mem_next_obss.shape}, test_mem_rewards: {test_mem_rewards.shape}')

        delta_obss = (next_obss - obss) / self.obss_abs_max
        delta_mem_obss = (mem_next_obss - mem_obss) / self.obss_abs_max

        inputs = np.concatenate((obss, actions), axis=-1)
        mem_inputs = np.concatenate((mem_obss, mem_actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        mem_targets = np.concatenate((delta_mem_obss, mem_rewards), axis=-1)

        delta_test_obss = (test_next_obss - test_obss)
        delta_test_mem_obss = (test_mem_next_obss - test_mem_obss) / self.obss_abs_max 

        test_inputs = np.concatenate((test_obss, test_actions), axis=-1)
        test_mem_inputs = np.concatenate((test_mem_obss, test_mem_actions), axis=-1)
        test_targets = np.concatenate((delta_test_obss, test_rewards), axis=-1)
        test_mem_targets = np.concatenate((delta_test_mem_obss, test_mem_rewards), axis=-1)
        
        return inputs, targets, terminals, mem_inputs, mem_targets, test_inputs, test_targets, test_mem_inputs, test_mem_targets

    def train(
        self,
        data: Dict,
        full_dataset: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        # max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        use_tqdm: bool = False,
    ) -> None:
        self.use_tqdm = use_tqdm
        train_inputs, train_targets, train_terminals, train_mem_inputs, train_mem_targets, holdout_inputs, holdout_targets, holdout_mem_inputs, holdout_mem_targets = self.format_samples_for_training(data, full_dataset)

        train_inputs_unscaled = deepcopy(train_inputs)
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        train_mem_inputs = self.scaler.transform(train_mem_inputs)
        holdout_mem_inputs = self.scaler.transform(holdout_mem_inputs)

        trainset = MemMultiStepDataset(train_inputs, train_targets, train_mem_inputs, train_mem_targets, train_terminals, self.train_horizon, train_inputs_unscaled)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        holdoutset = MemDataset(holdout_inputs, holdout_targets, holdout_mem_inputs, holdout_mem_targets)
        holdoutloader = torch.utils.data.DataLoader(holdoutset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        epoch = 0
        holdout_loss = 1e10
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            t0 = time.time()
            train_loss = self.train_model_epoch(epoch, trainloader)
            logger.logkv("loss/train_time", time.time() - t0)
            t1 = time.time()
            new_holdout_loss = self.test_model(holdoutloader)
            logger.logkv("loss/holdout_time", time.time() - t1)
            
            if new_holdout_loss <= holdout_loss:
                holdout_loss = new_holdout_loss
                
            self.save(epoch, logger.model_dir)
            
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.logkv("beta", self.beta)
            logger.dumpkvs(exclude=["policy_training_progress"])
            
            if (max_epochs and (epoch >= max_epochs)): 
                break

    def train_model_epoch(self, epoch, trainloader):
        t0 = time.time()

        train_losses = []
        self.model.train()
        loader = tqdm(trainloader) if self.use_tqdm else trainloader
        for batch_idx, batch in enumerate(loader):
    
            step = batch_idx if epoch == 1 else (epoch-1)*self.num_batches + batch_idx
            self.beta = 0
            if self.len_of_beta_step != 0 and self.num_beta_steps != 0:
                self.beta = max(0.99 * (1 - int(step / self.len_of_beta_step)/self.num_beta_steps), 0)
            
            train_loss = self.train_model_step(batch)
            
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            self.diagnostics["beta/train"] = self.beta
        self.num_batches = batch_idx + 1
        return np.mean(train_losses)

    def train_model_step(self, batch):    
        inputs, list_of_unscaled_actions, list_of_targets, mem_inputs, mem_targets, list_of_mem_inputs, list_of_mem_targets = batch
        inputs, mem_inputs, mem_targets = inputs.to(self.device), mem_inputs.to(self.device), mem_targets.to(self.device)
        list_of_unscaled_actions = [item.to(self.device) for item in list_of_unscaled_actions]
        list_of_targets = [item.to(self.device) for item in list_of_targets]
        list_of_mem_inputs = [item.to(self.device) for item in list_of_mem_inputs]
        list_of_mem_targets = [item.to(self.device) for item in list_of_mem_targets]

        loss = 0.0
        for t in range(self.train_horizon):
            if t == 0:
                current_inputs = inputs
                current_mem_inputs, current_mem_targets = mem_inputs, mem_targets
                with torch.no_grad():
                    current_dist = self.compute_distances(current_inputs, current_mem_inputs)
            else:                
                # ############# New
                # obs_dim = preds.shape[1] - 1
                # current_inputs = torch.cat([next_obss, list_of_unscaled_actions[t]], dim=-1)
                # with torch.no_grad():
                #     current_mem_inputs, current_mem_targets = self.find_memories(current_inputs)
                #     current_mem_inputs = self.scaler.transform_tensor(current_mem_inputs)
                # current_inputs = self.scaler.transform_tensor_with_tensor(current_inputs)
                # with torch.no_grad():
                #     current_dist = self.compute_distances(current_inputs, current_mem_inputs)

                ############### Alt
                current_inputs = torch.cat([next_obss, list_of_unscaled_actions[t]], dim=-1)
                current_inputs = self.scaler.transform_tensor_with_tensor(current_inputs)
                current_mem_inputs, current_mem_targets = list_of_mem_inputs[t], list_of_mem_targets[t]
                with torch.no_grad():
                    current_dist = self.compute_distances(current_inputs, current_mem_inputs)

            preds = self.model(
                inputs=current_inputs,
                mem_targets=current_mem_targets,
                dist=current_dist,
                beta=self.beta,
            )
            # print(t, current_inputs.min(), current_inputs.max())
            # print(t, current_mem_targets.min(), current_mem_targets.max())
            # print(t, current_dist.min(), current_dist.max())

            loss = loss + F.mse_loss(preds, list_of_targets[t]) 

            next_obss = preds[:, :-1] * self.obss_abs_max_tensor
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def test_model(self, testloader):
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            loader = tqdm(testloader) if self.use_tqdm else testloader
            for batch in loader:
                test_loss = self.test_model_step(batch)
                test_losses.append(test_loss)
        return np.mean(test_losses)

    def test_model_step(self, batch):
        new_batch = [item.to(self.device) for item in batch]
        inputs, targets, mem_inputs, mem_targets = new_batch

        dist = self.compute_distances(inputs, mem_inputs)

        preds = self.model(
            inputs=inputs,
            mem_targets=mem_targets,
            dist=dist,
            beta=self.beta, 
        )

        # unnormalize predicted next states
        preds[..., :-1] = preds[..., :-1] * self.obss_abs_max_tensor


        loss = F.mse_loss(preds, targets) 

        return loss.detach().cpu().item()
    
    def compute_distances(self, inputs, mem_inputs, dim=1, pnorm=2):
        dist = torch.norm(inputs - mem_inputs, p=pnorm, dim=dim).unsqueeze(1)

        mean_dists = torch.mean(dist); max_dists = torch.max(dist); min_dists = torch.min(dist)
        self.diagnostics[f'dist/train/mean'] = mean_dists.item()
        self.diagnostics[f'dist/train/max'] = max_dists.item()
        self.diagnostics[f'dist/train/min'] = min_dists.item()
            
        return dist


    def save(self, epoch, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
        if epoch % 50 == 0:
            torch.save(self.model.state_dict(), os.path.join(save_path, f"dynamics_{epoch}.pth"))

    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)