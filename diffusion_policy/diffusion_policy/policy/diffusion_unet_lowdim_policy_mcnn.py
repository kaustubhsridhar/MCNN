from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os, pickle 

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            num_memories_frac=0.1,
            lamda=10.0,
            Lipz=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.lamda = lamda
        self.Lipz = Lipz
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        loc = f'../mems_obs/updated_datasets/kitchen/updated_{num_memories_frac}_frac.pkl'
        if os.path.exists(loc):
            with open(loc, 'rb') as f:
                updated_data = pickle.load(f)
            # loading set of all unique memories
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.memories_obs = torch.as_tensor(updated_data['memories_obs'], device=device)
            self.memories_act = torch.as_tensor(updated_data['memories_act'], device=device)
        else:
            raise ValueError(f"Memories don't exist at {loc}.")
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            nobs=None,
            nmem_obs=None,
            nmem_act=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            neural_net_pred = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            with torch.no_grad():
                dist = torch.norm(nobs - nmem_obs, p=2, dim=-1).unsqueeze(-1)
            exp_lamda_dist = torch.exp(- self.lamda * dist)
            exp_lamda_dist = torch.cat([exp_lamda_dist, torch.zeros((exp_lamda_dist.shape[0], neural_net_pred.shape[1]-exp_lamda_dist.shape[1], 1), device=dist.device)], dim=1) # padding with 0 dist
            beta = 0
            pred = nmem_act * exp_lamda_dist + self.Lipz * (1.0 - exp_lamda_dist) * self.shifted_crazy_relu(neural_net_pred, beta)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                pred, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet

        # search for memory
        obs = obs_dict['obs'] # (B, To=2, Do=60)
        mem_observations, mem_actions = [], []
        for i in range(obs.shape[1]):
            _, closest_nodes = torch.cdist(obs[:, i, :], self.memories_obs).min(dim=-1)
            mem_observations.append( self.memories_obs[closest_nodes, :] )
            mem_actions.append( self.memories_act[closest_nodes, :] )
        mem_observations = torch.stack(mem_observations, dim=1) # (B, To=2, Do=60)
        mem_actions = torch.stack(mem_actions, dim=1) # (B, To=2, Da=9)

        # continued
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        nmem_obs = self.normalizer['obs'].normalize(mem_observations)
        nmem_act = self.normalizer['action'].normalize(mem_actions)
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # expand
        if nmem_act.shape[1] < T:
            nmem_act = torch.cat([nmem_act, torch.zeros((B, T-nmem_act.shape[1], Da), device=obs.device)], dim=1) # (B, T=16, Da=9)

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            nobs=nobs,
            nmem_obs=nmem_obs,
            nmem_act=nmem_act,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs'] # (B, T=16, Do=60)
        action = nbatch['action'] # (B, T=16, Da=9)
        mem_observation = nbatch['mem_observations'] # (B, T=16, Do=60)
        mem_action = nbatch['mem_actions'] # (B, T=16, Da=9)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual or full sample
        neural_net_pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond) # (B, T, D)
        with torch.no_grad():
            dist = torch.norm(obs - mem_observation, p=2, dim=-1).unsqueeze(-1) # (B, T, 1)
        exp_lamda_dist = torch.exp(- self.lamda * dist)
        beta = 0
        pred = mem_action * exp_lamda_dist + self.Lipz * (1.0 - exp_lamda_dist) * self.shifted_crazy_relu(neural_net_pred, beta) # (B, T, D)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory # (B, T, D)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
