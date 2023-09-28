import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from copy import deepcopy
import time 


def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)

class MemEnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        dataset: Dict = {},
        Lipz: float = 100.0,
        lamda: float = 1.0,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.Lipz = Lipz
        self.lamda = lamda
        self.memory_act = self.shifted_crazy_relu
        self.len_of_beta_step = 10
        self.num_beta_steps = 10

        self.memories_obs = torch.as_tensor(dataset["memories_obs"]).to(self.model.device)
        self.memories_actions = torch.as_tensor(dataset["memories_actions"]).to(self.model.device)
        self.memories_next_obs = torch.as_tensor(dataset["memories_next_obs"]).to(self.model.device)
        self.memories_rewards = torch.as_tensor(dataset["memories_rewards"]).to(self.model.device).unsqueeze(1)
        self.memories_inputs = torch.cat([self.memories_obs, self.memories_actions], dim=1)

        self.obss_abs_max = np.max(np.abs(dataset["observations"]), axis=0, keepdims=True)
        self.obss_abs_max_tensor = torch.as_tensor(self.obss_abs_max).to(self.model.device)

        self.diagnostics = {}

    def shifted_sigmoid(self, x, beta):
        return 2 * nn.Sigmoid()(x) - 1

    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1
    
    def find_memories(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, closest_nodes = torch.cdist(inputs, self.memories_inputs).min(dim=1)
        mem_obss = self.memories_obs[closest_nodes, :]
        mem_actions = self.memories_actions[closest_nodes, :]
        mem_next_obss = self.memories_next_obs[closest_nodes, :]
        mem_rewards = self.memories_rewards[closest_nodes, :]

        delta_mem_obss = (mem_next_obss - mem_obss) / self.obss_abs_max_tensor

        mem_inputs = torch.cat([mem_obss, mem_actions], dim=1)
        mem_targets = torch.cat([delta_mem_obss, mem_rewards], dim=1)
        return mem_inputs, mem_targets

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        raw_mean, logvar = self.model(obs_act)


        # Wrapper
        inputs = torch.as_tensor(deepcopy(obs_act)).to(self.model.device)
        mem_inputs, mem_targets = self.find_memories(inputs)
        inputs = self.scaler.transform_tensor(inputs)
        mem_inputs = self.scaler.transform_tensor(mem_inputs)
        
        dist = self.compute_distances(inputs, mem_inputs, dim=1)
        mean = self.wrapper(raw_mean=raw_mean, dist=dist, mem_targets=mem_targets, beta=0)

        # unnormalize predicted next states
        mean[..., :-1] = mean[..., :-1] * self.obss_abs_max_tensor

        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    # @ torch.no_grad()
    # def sample_next_obss(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     num_samples: int
    # ) -> torch.Tensor:
    #     obs_act = torch.cat([obs, action], dim=-1)
    #     obs_act = self.scaler.transform_tensor(obs_act)
    #     mean, logvar = self.model(obs_act)
    #     mean[..., :-1] += obs
    #     std = torch.sqrt(torch.exp(logvar))

    #     mean = mean[self.model.elites.data.cpu().numpy()]
    #     std = std[self.model.elites.data.cpu().numpy()]

    #     samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
    #     next_obss = samples[..., :-1]
    #     return next_obss

    def format_samples_for_training(self, data: Dict, full_dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
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
        
        return inputs, targets, mem_inputs, mem_targets, test_inputs, test_targets, test_mem_inputs, test_mem_targets

    def train(
        self,
        data: Dict,
        full_dataset: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        # max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        use_tqdm: bool = False,
    ) -> None:
        train_inputs, train_targets, train_mem_inputs, train_mem_targets, holdout_inputs, holdout_targets, holdout_mem_inputs, holdout_mem_targets = self.format_samples_for_training(data, full_dataset)
        train_size = train_inputs.shape[0]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        train_mem_inputs = self.scaler.transform(train_mem_inputs)
        holdout_mem_inputs = self.scaler.transform(holdout_mem_inputs)
        
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            t0 = time.time()
            train_loss = self.learn(epoch, train_inputs[data_idxes], train_targets[data_idxes], train_mem_inputs[data_idxes], train_mem_targets[data_idxes], batch_size, logvar_loss_coef)
            logger.logkv("loss/train_time", time.time() - t0)
            t1 = time.time()
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets, holdout_mem_inputs, holdout_mem_targets)
            logger.logkv("loss/holdout_time", time.time() - t1)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.logkv("beta", self.beta)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            self.save(epoch, logger.model_dir)
            
            if (max_epochs and (epoch >= max_epochs)): # (cnt >= max_epochs_since_update) or 
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(epoch, logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))

    def wrapper(self, raw_mean, dist, mem_targets, beta):
        lamda_in_exp = self.lamda * 10 # self.lamda * self.Lipz * 10
        exp_lamda_dist = torch.exp(-lamda_in_exp * dist)
        mean = mem_targets[..., :-1] * exp_lamda_dist + self.Lipz * (1.0-exp_lamda_dist) * self.memory_act(raw_mean[..., :-1], beta)
        mean_and_reward = torch.cat((mean, raw_mean[..., -1:]), dim=-1)
        return mean_and_reward
    
    def learn(
        self,
        epoch: int,
        inputs: np.ndarray,
        targets: np.ndarray,
        mem_inputs: np.ndarray,
        mem_targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_batch = torch.as_tensor(inputs_batch).to(self.model.device)
            mem_inputs_batch = mem_inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            mem_inputs_batch = torch.as_tensor(mem_inputs_batch).to(self.model.device)
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            mem_targets_batch = mem_targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            mem_targets_batch = torch.as_tensor(mem_targets_batch).to(self.model.device)
            
            raw_mean, logvar = self.model(inputs_batch)

            
            # Wrapper
            with torch.no_grad():
                dist = self.compute_distances(inputs_batch, mem_inputs_batch)

            step = batch_num if epoch == 1 else (epoch-1)*self.num_batches + batch_num
            self.beta = 0
            if self.len_of_beta_step != 0 and self.num_beta_steps != 0:
                self.beta = max(0.99 * (1 - int(step / self.len_of_beta_step)/self.num_beta_steps), 0)

            mean = self.wrapper(raw_mean=raw_mean, dist=dist, mem_targets=mem_targets_batch, beta=self.beta)
            

            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        self.num_batches = batch_num + 1
        return np.mean(losses)
    
    def compute_distances(self, inputs, mem_inputs, dim=2, pnorm=2):
        dist = torch.norm(inputs - mem_inputs, p=pnorm, dim=dim).unsqueeze(dim)

        mean_dists = torch.mean(dist); max_dists = torch.max(dist); min_dists = torch.min(dist)
        self.diagnostics[f'dist/train/mean'] = mean_dists.item()
        self.diagnostics[f'dist/train/max'] = max_dists.item()
        self.diagnostics[f'dist/train/min'] = min_dists.item()
            
        return dist
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray, mem_inputs: np.ndarray, mem_targets: np.ndarray) -> List[float]:
        self.model.eval()
        inputs = torch.as_tensor(inputs).to(self.model.device)
        targets = torch.as_tensor(targets).to(self.model.device)
        mem_inputs = torch.as_tensor(mem_inputs).to(self.model.device)
        mem_targets = torch.as_tensor(mem_targets).to(self.model.device)
        raw_mean, _ = self.model(inputs)


        # Wrapper
        with torch.no_grad():
            dist = self.compute_distances(inputs, mem_inputs, dim=1)
        
        mean = self.wrapper(raw_mean=raw_mean, dist=dist, mem_targets=mem_targets, beta=0)

        # unnormalize predicted next states
        mean[..., :-1] = mean[..., :-1] * self.obss_abs_max_tensor
        

        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, epoch: int, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
        if epoch % 50 == 0:
            torch.save(self.model.state_dict(), os.path.join(save_path, f"dynamics_{epoch}.pth"))
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
