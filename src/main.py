from enviroment import RoadEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common import env_checker
import reward
import argparse
import optuna
from optuna_dashboard import run_server
import matplotlib.pyplot as plt
import numpy as np

MAX_STEPS = 1_000_000


def tune_PPO(trial):
    env = RoadEnv(reward.reward, render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = trial.suggest_float("lr", 0.00001, 0.01)
    cliprange = trial.suggest_float("cliprange", 0.1, 0.7)
    gamma = trial.suggest_float("gamma", 0.5, 0.95)
    model = PPO(
        "MultiInputPolicy", env, clip_range=cliprange, learning_rate=lr, gamma=gamma
    ).learn(total_timesteps=1000000)
    avg_ret = mean(env.avg_returns)
    return avg_ret


def train_ppo(load_model=False):
    env = RoadEnv(reward.reward, render_mode=None)
    if load_model:
        ppo = PPO.load("models/baselines-ppo")
    else:
        ppo = PPO("MultiInputPolicy", env, verbose=1)
    ppo = ppo.learn(total_timesteps=MAX_STEPS)
    ppo.save("models/baselines-ppo")
    return env.avg_rewards


def train_dqn(load_model=False):
    env = RoadEnv(reward.reward, render_mode=None)
    if load_model:
        dqn = DQN.load("models/baselines-dqn")
    else:
        dqn = DQN("MultiInputPolicy", env, verbose=1)
    dqn = dqn.learn(total_timesteps=MAX_STEPS)
    dqn.save("models/baselines-ppo")
    return env.avg_rewards


def main(render):
    # Parallel environments
    # vec_env = make_vec_env(env, n_envs=4)

    # TODO hyperparameter tune

    ppo_rews = train_ppo()
    dqn_rews = train_dqn()

    plt.title("DQN vs PPO episode rewards")
    plt.plot(np.arange(len(ppo_rews)), ppo_rews, label="PPO")
    plt.plot(np.arange(len(dqn_rews)), dqn_rews, label="DQN")
    plt.legend()
    plt.show()

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


    """


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
