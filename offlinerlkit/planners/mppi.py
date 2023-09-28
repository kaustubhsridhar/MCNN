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
from offlinerlkit.planners.mppi_test import test

def tune(eval_env, dynamics, policy, device, init_horizon, eval_method='score'):
    ts = time.time()
    print(f'{eval_method=}')

    def dynamics_fn(x, u):
        x = x.cpu().numpy()
        u = u.cpu().numpy()
        next_x = torch.tensor(dynamics.step(x, u)[0], device=device).float()
        return next_x
    def running_cost_fn(x, u):
        x = x.cpu().numpy()
        u = u.cpu().numpy()
        r = torch.tensor(dynamics.step(x, u)[1], device=device).float().flatten()
        return -r
    
    obs_dim = np.prod(eval_env.observation_space.shape)
    min_action = torch.tensor(eval_env.action_space.low, device=device).float()
    max_action = torch.tensor(eval_env.action_space.high, device=device).float()
    noise_sigma = torch.eye(len(min_action), device=device).float() * 1
    print(f'\n{obs_dim=}, {min_action=}, {max_action=}, {noise_sigma.shape=}, {device=}\n')
   
    # create MPPI with some initial parameters
    mppi = MPPI(dynamics=dynamics_fn, 
                running_cost=running_cost_fn, 
                nx=obs_dim,
                noise_sigma=noise_sigma,
                num_samples=500,
                horizon=init_horizon, 
                device=device,
                u_min=min_action,
                u_max=max_action,
                lambda_=1e-2)
    
    ## Usage!
    # obs = eval_env.reset().reshape(1, -1)
    # action = mppi.command(obs)
    # print(action)
                

    """
    We then need to create an evaluation function for the tuner to tune on. 
    It should take no arguments and output a EvaluationResult populated at least by csts. 
    If you don't need rollouts for the cost evaluation, then you can set it to None in the return. 
    """
    _eval_episodes = 1
    def evaluate():
        t0 = time.time()
        rewards = []
        # diffs = []
        lengths = []
        rollouts = []

        num_episodes = 0
        episode_reward, episode_length = 0, 0
        # episode_diff = 0
        episode_rollout = []
        obs = eval_env.reset()
        mppi_time = 0

        while num_episodes < _eval_episodes:
            obs = obs.reshape(1, -1)
            tp = time.time()
            action = mppi.command(obs).cpu().numpy()
            mppi_time += time.time() - tp
            
            next_obs, reward, terminal, _ = eval_env.step(action)
            episode_reward += reward
            # if eval_method == 'imitation':
            #     model_free_action = policy.select_action(obs).flatten()
            #     assert model_free_action.shape == action.shape, f'{model_free_action.shape=}, {action.shape=}'
            #     diff = np.mean((action - model_free_action)**2)
            #     episode_diff += diff

            episode_length += 1
            episode_rollout.append(deepcopy(obs))

            obs = next_obs

            if terminal:
                rewards.append(episode_reward)
                # if eval_method == 'imitation':
                #     diffs.append(episode_diff)
                lengths.append(episode_length)
                rollouts.append(episode_rollout)

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                # episode_diff = 0
                episode_rollout = []
                obs = eval_env.reset()

        print(f'Evaluation took {time.time() - t0:.2f}s out which mppi took {mppi_time:.2f}s')

        # normalized_scores = [eval_env.get_normalized_score(R) for R in rewards]
        rewards = torch.tensor(rewards, device=device).float()
        # diffs = torch.tensor(diffs, device=device).float()
        # print(f'rewards: {rewards}')
        # print(f'normalized scores: {normalized_scores}')
        # print(f'diffs: {diffs}')

        # if eval_method == 'imitation':
        #     costs = diffs - rewards
        # else:
        costs = - rewards
        # print(f'costs: {costs} \n')
        rollouts = torch.tensor(rollouts, device=device).float()

        return autotune.EvaluationResult(costs, rollouts)

    # tuning with ray
    params_to_tune = ['horizon', 'sigma', 'lambda'] # 
    # create a tuner with a CMA-ES optimizer
    tuner = autotune.AutotuneMPPI(mppi, params_to_tune, evaluate_fn=evaluate, optimizer=autotune.CMAESOpt(sigma=1.0))
    # tune parameters for a number of iterations
    iterations = 100
    for i in range(iterations):
        print(f'iter: {i}')
        # results of this optimization step are returned
        res = tuner.optimize_step()
        print(f'optimization step complete')
        # get best results
        res = tuner.get_best_result()
        print(f'{res=}')
        print(f'{res.costs=}')
        normalized_scores = [eval_env.get_normalized_score(-c) for c in res.costs]
        print(f'{normalized_scores=}')
    # get best results and apply it to the controller
    # (by default the controller will take on the latest tuned parameter, which may not be best)
    res = tuner.get_best_result()
    tuner.apply_parameters(res.params)
    print(f'final {res=}')
    print(f'Tuning took {time.time() - ts:.2f}s')