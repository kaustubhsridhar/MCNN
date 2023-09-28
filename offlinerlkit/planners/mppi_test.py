"""
Credits to https://github.com/UM-ARM-Lab/pytorch_mppi
"""

import torch
import torch.nn.functional as F
from pytorch_mppi import MPPI, autotune, autotune_global
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
import numpy as np
from copy import deepcopy
import time


def test(eval_env, 
         dynamics, 
         vanilla_dynamics,
         device,
         num_samples=10000,
         horizon=6,
         sigma=[1.0]*6, # [7.8559e+00, 1.0000e-04, 5.7373e+00, 1.0000e-04, 5.4301e+00, 1.0000e-04],
         lambda_=0.51, # 0.0001,
        ):
    ts = time.time()

    def dynamics_fn(x, u):
        x = x.cpu().numpy()
        u = u.cpu().numpy()
        next_x = torch.tensor(dynamics.step(x, u)[0], device=device).float()
        next_x_vanilla = torch.tensor(vanilla_dynamics.step(x, u)[0], device=device).float()
        # print(f'{F.mse_loss(next_x, next_x_vanilla)=}')
        return next_x
    def running_cost_fn(x, u):
        x = x.cpu().numpy()
        u = u.cpu().numpy()
        r = torch.tensor(dynamics.step(x, u)[1], device=device).float().flatten()
        r_vanilla = torch.tensor(vanilla_dynamics.step(x, u)[1], device=device).float().flatten()
        # print(f'{F.mse_loss(r, r_vanilla)=}')
        return -r
    
    obs_dim = np.prod(eval_env.observation_space.shape)
    min_action = torch.tensor(eval_env.action_space.low, device=device).float()
    max_action = torch.tensor(eval_env.action_space.high, device=device).float()
    noise_sigma = torch.diag(torch.tensor(sigma, device=device).float())

    print(f'\n{obs_dim=}, {min_action=}, {max_action=}, {noise_sigma.shape=}, {device=}\n')

    # create MPPI with tuned parameters!
    mppi = MPPI(dynamics=dynamics_fn, 
                running_cost=running_cost_fn, 
                nx=obs_dim,
                noise_sigma=noise_sigma,
                num_samples=num_samples,
                horizon=horizon, 
                device=device,
                u_min=min_action,
                u_max=max_action,
                lambda_=lambda_)
    
    # Test:
    _eval_episodes = 3
    def evaluate():
        t0 = time.time()
        costs = []
        lengths = []
        rollouts = []

        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_rollout = []
        obs = eval_env.reset()
        mppi_time = 0

        while num_episodes < _eval_episodes:
            obs = obs.reshape(1, -1)
            tp = time.time()
            action = mppi.command(obs)
            mppi_time += time.time() - tp
            next_obs, reward, terminal, _ = eval_env.step(action.cpu().numpy())
            episode_reward += reward
            episode_length += 1
            episode_rollout.append(deepcopy(obs))

            obs = next_obs

            if terminal:
                costs.append(-episode_reward)
                lengths.append(episode_length)
                rollouts.append(episode_rollout)

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                episode_rollout = []
                obs = eval_env.reset()

        print(f'Evaluation took {time.time() - t0:.2f}s out which mppi took {mppi_time:.2f}s')
        costs = torch.tensor(costs, device=device).float()
        rollouts = torch.tensor(rollouts, device=device).float()
        print(f'{costs=}')
        normalized_scores = [eval_env.get_normalized_score(-c.item()) for c in costs]

        return np.mean(normalized_scores), np.std(normalized_scores)
    
    mean_score, std_score = evaluate()
    print(f'Score {mean_score:.2f} +- {std_score:.2f}')
    print(f'Evaluation took {time.time() - ts:.2f}s')