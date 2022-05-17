import os
import time

import gym
import numpy as np
import torch

import utils
from algorithms.factory import make_agent
from arguments import parse_args
from env.wrappers import make_feat_env
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, num_episodes, video, video_name):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        video.init(enabled=(i == 0))
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        video.save(f"{video_name}.mp4")


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)

    env = make_feat_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode="train",
    )

    video_dir = utils.make_dir(os.path.join("./", "video"))
    video = VideoRecorder(video_dir if True else None, height=448, width=448)
    agent = torch.load(args.checkpoint)
    evaluate(env, agent, args.eval_episodes, video, args.video_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
