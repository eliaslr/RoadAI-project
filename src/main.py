from enviroment import RoadEnv
from ppo import PPO

import reward
import matplotlib.pyplot as plt
import argparse

# import hydra


def main(render):
    # ray.init()
    env = RoadEnv(reward.reward, render_mode=render)
    ppo = PPO(env, 0.001, 0.2, "models/ppo/")
    ppo.train()
    # Show an episode to see how the system performs
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # ppo.train(render_mode=render)
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # Show Metrics

import optuna
def objective(trial):
    render = parse_args()

    lr = trial.suggest_float('lr', 0, 1)
    cr = trial.suggest_float("cliprange", 0, 1)
    g  = trial.suggest_float("gamma", 0, 1)

    env = RoadEnv(reward.reward, render_mode=render)
    ppo = PPO(env,
              lr = lr,
              cliprange = cr,
              model_path = "models/ppo/",
              load_model=True,
              gamma=g)
    return ppo.env.avg_rewards


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
    study = optuna.create_study()
    study.optimize(objective, n_trials = 10)

    study.best_params
    # main(args.render)