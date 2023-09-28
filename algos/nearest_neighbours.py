import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs_td3bc
from offlinerlkit.policy_trainer import NearestNeighboursEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="vinn", choices=["1nn", "vinn"])
    parser.add_argument("--task", type=str, default="pen-human-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument('--chosen-percentage', type=float, default=1.0, choices=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--use-tqdm', type=int, default=1) # 1 or 0

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset_percentbc(args.task, args.chosen_percentage, args.num_memories_frac)
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
    # obs_mean, obs_std = buffer.normalize_obs()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

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

    # log
    record_params = ["chosen_percentage"]
    log_dirs = make_log_dirs_td3bc(task_name=args.task, chosen_percentage=args.chosen_percentage, algo_name=args.algo_name, seed=args.seed, args=vars(args), record_params=record_params)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    evaluator = NearestNeighboursEvaluator(
        algo=args.algo_name,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        device=args.device,
        perception_model=perception_model,
        use_tqdm=args.use_tqdm
    )
    evaluator.evaluate()


if __name__ == "__main__":
    train()