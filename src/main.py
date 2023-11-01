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
import pickle

MAX_EPS = 250
MAX_STEPS = MAX_EPS * 5000


def tune_PPO(trial):
    env = RoadEnv(reward.reward, render_mode=None)
    lr = trial.suggest_float("lr", 0.00001, 0.01)
    cliprange = trial.suggest_float("cliprange", 0.1, 0.7)
    gamma = trial.suggest_float("gamma", 0.5, 0.95)
    model = PPO(
        "MultiInputPolicy", env, clip_range=cliprange, learning_rate=lr, gamma=gamma
    ).learn(total_timesteps=1_000_000)
    avg_ret = np.mean(env.avg_rewards)
    return avg_ret


def train_ppo(load_model=False):
    env = RoadEnv(reward.reward, render_mode=None)
    if load_model:
        ppo = PPO.load("models/baselines-ppo")
    else:
        ppo = PPO(
            "MultiInputPolicy", env, learning_rate=0.0001, clip_range=0.20, gamma=0.90
        )
    ppo = ppo.learn(total_timesteps=MAX_STEPS)
    ppo.save("models/baselines-ppo")
    return env.avg_rewards


def train_dqn(load_model=False):
    env = RoadEnv(reward.reward, render_mode=None)
    if load_model:
        dqn = DQN.load("models/baselines-dqn")
    else:
        dqn = DQN("MultiInputPolicy", env)
    dqn = dqn.learn(total_timesteps=MAX_STEPS)
    dqn.save("models/baselines-dqn")
    return env.avg_rewards


def show_models():
    env = RoadEnv(reward.reward, render_mode="pygame")
    model = PPO.load("models/baselines-ppo")
    # model = DQN.load("models/baselines-dqn")
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, _, _, _ = env.step(action)


def make_plot(ppo_rews, dqn_rews):
    plt.title(f"PPOvsDQN {MAX_EPS} episode rewards")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average episodic reward")
    plt.plot(np.arange(len(ppo_rews)), ppo_rews, label="PPO")
    plt.plot(np.arange(len(dqn_rews)), dqn_rews, label="DQN")
    plt.legend()
    plt.savefig(f"graphs/PPOvsDQN{MAX_EPS}.png")
    plt.show()


def main(render):
    """
    ppo_rews = train_ppo()
    with open(f"results/ppo{MAX_EPS}.pkl", "wb") as file:
        pickle.dump(ppo_rews, file)
    """
    """
    with open("results/ppo1000.pkl", "rb") as file:
        ppo_rews = pickle.load(file)
    with open("results/dqn1000.pkl", "rb") as file:
        dqn_rews = pickle.load(file)
    dqn_rews = train_dqn()
    with open(f"results/dqn{MAX_EPS}.pkl", "wb") as file:
        pickle.dump(dqn_rews, file)
    """
    # make_plot(ppo_rews, dqn_rews)
    show_models()

    """
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction="maximize", storage=storage)
    study.optimize(tune_PPO, n_trials=30)
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
