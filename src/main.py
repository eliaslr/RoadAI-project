from enviroment import RoadEnv
from ppo import PPO
import torch

import reward
import matplotlib.pyplot as plt
import argparse

# import hydra


def main(render):
    # ray.init()
    env = RoadEnv(reward.reward, render_mode=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo = PPO(env, 0.001, 0.2, "models/ppo/", device = device)
    ppo.train()
    # Show an episode to see how the system performs
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # ppo.train(render_mode=render)
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs

    # Show Metrics


def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Train roadAI MARL.")
    parser.add_argument(
        "-r",
        "--render",
        help="How to render the model while training (console, pygame). Leave blank to not render",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.render)
