from enviroment import RoadEnv
from ppo import PPO
import torch

import reward
import argparse
import optuna
from optuna_dashboard import run_server


def tune(trial):
    env = RoadEnv(reward.reward, render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_a = trial.suggest_float("lr_a", 0.00001, 0.01)
    lr_c = trial.suggest_float("lr_c", 0.00001, 0.01)
    cliprange = trial.suggest_float("cliprange", 0.1, 0.7)
    gamma = trial.suggest_float("gamma", 0.5, 0.95)
    ppo = PPO(
        env,
        lr_a=lr_a,
        lr_c=lr_c,
        cliprange=cliprange,
        gamma=gamma,
        model_path="models/ppo/",
        device=device,
    )
    avg_ret = ppo.train()
    return avg_ret


def main(render):
    env = RoadEnv(reward.reward, render_mode=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = PPO(
        env,
        lr_a=0.001,
        lr_c=0.001,
        gamma=0.85,
        cliprange=0.2,
        model_path="models/ppo/",
        device=device,
    )
    ppo.train()
    """
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction="maximize", storage=storage)
    study.optimize(tune, n_trials=30)
    run_server(storage)
    # Show an episode to see how the system performs
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # ppo.train(render_mode=render)
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs

    """
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
