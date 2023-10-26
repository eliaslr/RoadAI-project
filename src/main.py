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
    cliprange = trial.suggest_float("cliprange", 0.1, 0.5)
    gamma = trial.suggest_float("gamma", 0.8, 0.95)
    ppo = PPO(
        env,
        lr_a=lr_a,
        lr_c=lr_c,
        cliprange=cliprange,
        gamma=gamma,
        model_path="models/ppo/",
        device=device
    )
    avg_ret = ppo.train()
    return avg_ret


def main(render):
    """
    env = RoadEnv(reward.reward, render_mode=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = PPO(
        env,
        lr_a=0.001,
        lr_c=0.001,
        epsilon_start=0.6,
        epsilon_decay=0.03,
        cliprange=0.2,
        model_path="models/ppo/",
        device=device,
    )
    ppo.train()
    """
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction="maximize", storage=storage)
    study.optimize(tune, n_trials=1000)
    run_server(storage)

    # Show an episode to see how the system performs
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # ppo.train(render_mode=render)
    # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs
    # Show Metrics

# import optuna
# def objective(trial):
#     render = parse_args()

#     lr = trial.suggest_float('lr', 0, 1)
#     cr = trial.suggest_float("cliprange", 0, 1)
#     g  = trial.suggest_float("gamma", 0, 1)

#     env = RoadEnv(reward.reward, render_mode=render)
#     ppo = PPO(env,
#               lr = lr,
#               cliprange = cr,
#               model_path = "models/ppo/",
#               load_model=True,
#               gamma=g)
#     return ppo.env.avg_rewards


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
    # study = optuna.create_study()
    # study.optimize(objective, n_trials = 10)

    # study.best_params
    args = parse_args()
    main(args.render)