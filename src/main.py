from enviroment import RoadEnv
from ppo import PPO

import reward
import matplotlib.pyplot as plt
import argparse

# import hydra


def main(render):
    # ray.init()
    env = RoadEnv(reward.reward)
    ppo = PPO(env, 0.0001, 0.2)
    rewards = []
    NUM_OF_TRAINING_EPS = 1
    # Train for X eps
    for _ in range(NUM_OF_TRAINING_EPS):
        ppo.train(render_mode="pygame")
        rewards.append(env.avg_rewards)
        # Show an episode to see how the system performs
        # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
        ppo.train(render_mode=render)
        # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs

    # Show Metrics

def parse_args():
  """Parse command line argument."""

  parser = argparse.ArgumentParser("Train roadAI MARL.")
  parser.add_argument("-r", "--render", help="How to render the model while training (console, pygame). Leave blank to not render")

  return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.render)
