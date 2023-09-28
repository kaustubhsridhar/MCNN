import argparse
import gym  
import d4rl

ibc_data = { # https://arxiv.org/pdf/2109.00137.pdf
    'pen-human-v1': (2586, 65),
    'hammer-human-v1': (-133, 26),
    'door-human-v0': (361, 67),
    'relocate-human-v0': (-0.1, 2.4),
}
for task, (mean, std) in ibc_data.items():
    env = gym.make(task)
    print(env.observation_space.shape)
    normalized_std = (std) / (env.ref_max_score - env.ref_min_score) * 100 # to normalize std we only need to divide by the range / 100; no subtraction in numerator
    print(f'\n', task, f'mean: ', env.get_normalized_score(mean) * 100, f'std: ', normalized_std, f'\n', )
