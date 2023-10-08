import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, Critic, MemActor
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs_td3bc
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import TD3BCPolicy, MemTD3BCPolicy


"""
suggested hypers
alpha=2.5 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mem_bc", choices=["bc", "mem_bc", "td3bc", "mem_td3bc"])
    parser.add_argument("--task", type=str, default="pen-human-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5) # is set to 0 for only BC below
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument('--chosen-percentage', type=float, default=1.0)#, choices=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--use-tqdm', type=int, default=1) # 1 or 0
    parser.add_argument('--use-random-memories', type=int, default=0) # 1 or 0
    parser.add_argument('--save_videos', action='store_true', default=False)

    return parser.parse_args()


def train(args=get_args()):
    # update args
    if args.algo_name in ["bc", "mem_bc"]:
        args.only_bc = True
        args.alpha = 0.0
    else:
        args.only_bc = False

    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset_percentbc(args.task, args.chosen_percentage, args.num_memories_frac, use_random_memories=args.use_random_memories)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    args.obs_shape = (512,) if 'carla' in args.task else env.observation_space.shape
    args.action_dim = 2 if 'carla' in args.task else np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_hidden_dims = [1024, 1024] if 'carla' in args.task else [256, 256]
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=actor_hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    if "mem" in args.algo_name:
        actor = MemActor(actor_backbone, args.action_dim, device=args.device, Lipz=args.Lipz, lamda=args.lamda)
    else:
        actor = Actor(actor_backbone, args.action_dim, device=args.device)

    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # load perception encoder if carla
    if 'carla' in args.task:
        from offlinerlkit.carla.carla_model import CoILICRA
        from offlinerlkit.carla.carla_config import MODEL_CONFIGURATION
        carla_model_state_dict = torch.load('data/models/nocrash/resnet34imnet10S1/checkpoints/660000.pth')['state_dict']
        carla_model = CoILICRA(MODEL_CONFIGURATION)
        carla_model.load_state_dict(carla_model_state_dict)
        carla_model.eval()
        def perception_model(obs):
            obs = obs.reshape(1, 48, 48, 3)
            obs = torch.tensor(obs).permute(0, 3, 1, 2).float()
            return carla_model.perception(obs)[0].detach().numpy()
        print(f'loaded carla_model')
    else:
        perception_model = None

    # create policy
    if "mem" in args.algo_name:
        policy = MemTD3BCPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            max_action=args.max_action,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            update_actor_freq=args.update_actor_freq,
            alpha=args.alpha,
            scaler=scaler,
            only_bc=args.only_bc,
            dataset=dataset,
            perception_model=perception_model,
        )
    else:
        policy = TD3BCPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            max_action=args.max_action,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            update_actor_freq=args.update_actor_freq,
            alpha=args.alpha,
            scaler=scaler,
            only_bc=args.only_bc,
            perception_model=perception_model,
        )

    # log
    record_params = ["chosen_percentage"]
    if "mem" in args.algo_name:
        record_params += ["num_memories_frac", "Lipz", "lamda"]

    log_dirs = make_log_dirs_td3bc(task_name=args.task, chosen_percentage=args.chosen_percentage, algo_name=args.algo_name, seed=args.seed, args=vars(args), record_params=record_params)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        save_videos=args.save_videos,
        freq_save_videos=50, # save videos every 50 epochs
    )

    # train
    policy_trainer.train(use_tqdm=args.use_tqdm)


if __name__ == "__main__":
    train()