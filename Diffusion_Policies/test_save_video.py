import gym
import d4rl
import numpy as np
import torch
import skvideo.io
import os

env = gym.make('pen-human-v1')

for episode in range(10):
    obs = env.reset()
    done = False
    episode_reward = 0
    arrs = []

    while not done:
        # Here, I'm just taking random actions. Replace this with your policy if needed.
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        
        curr_frame = env.sim.render(width=640, height=480, mode='offscreen', camera_name=None, device_id=0)
        arrs.append(curr_frame[::-1, :, :])

        episode_reward += reward

    os.makedirs('./videos', exist_ok=True)
    skvideo.io.vwrite( f'./videos/episode_{episode+1}.mp4', np.asarray(arrs))
    print(f"Episode {episode + 1} reward: {episode_reward} saved video at ./videos/episode_{episode+1}.mp4")
env.close()