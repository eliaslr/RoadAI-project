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
import time

MAX_EPS = 1000
MAX_STEPS = MAX_EPS * 7500


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
            "MultiInputPolicy", env, learning_rate=0.001, clip_range=0.20, gamma=0.95
        )
    ppo = ppo.learn(total_timesteps=MAX_STEPS)
    ppo.save("models/baselines-ppo")
    return env.avg_rewards, env.avg_mass


def train_dqn(load_model=False):
    env = RoadEnv(reward.reward, render_mode=None)
    if load_model:
        dqn = DQN.load("models/baselines-dqn")
    else:
        dqn = DQN("MultiInputPolicy", env)
    dqn = dqn.learn(total_timesteps=MAX_STEPS)
    dqn.save("models/baselines-dqn")
    return env.avg_rewards, env.avg_mass


def show_models():
    env = RoadEnv(reward.reward, render_mode="pygame")
    model = PPO.load("models/baselines-ppo")
    # model = DQN.load("models/baselines-dqn")
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, _, _, _ = env.step(action)


def make_plot(ppo_rews, ppo_mass, dqn_rews, dqn_mass):
    plt.title(f"PPOvsDQN {MAX_EPS} episode rewards")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average episodic reward")
    ppo_mu = np.full(len(ppo_rews), np.convolve(ppo_rews, np.ones(5) / 5, mode="same"))
    dqn_mu = np.full(len(dqn_rews), np.convolve(dqn_rews, np.ones(5) / 5, mode="same"))
    plt.plot(ppo_mu, label="PPO", color="orange")
    plt.plot(dqn_mu, label="DQN", color="deepskyblue")
    plt.plot(np.arange(len(ppo_rews)), ppo_rews, alpha=0.2, color="orange")
    plt.plot(np.arange(len(dqn_rews)), dqn_rews, alpha=0.2, color="deepskyblue")
    plt.legend()
    plt.savefig(f"graphs/PPOvsDQN{MAX_EPS}rews.png")
    plt.show()
    plt.title(f"PPOvsDQN {MAX_EPS} Mass per episode")
    plt.xlabel("Number of episodes")
    plt.ylabel("Mass in kgs")
    ppo_mu = np.full(len(ppo_rews), np.convolve(ppo_mass, np.ones(5) / 5, mode="same"))
    dqn_mu = np.full(len(dqn_rews), np.convolve(dqn_mass, np.ones(5) / 5, mode="same"))
    plt.plot(ppo_mu, color="orange", label="PPO")
    plt.plot(dqn_mu, color="deepskyblue", label="DQN")
    plt.plot(np.arange(len(ppo_mass)), ppo_mass, alpha=0.2, color="orange")
    plt.plot(np.arange(len(dqn_mass)), dqn_mass, alpha=0.2, color="deepskyblue")
    plt.legend()
    plt.savefig(f"graphs/PPOvsDQN{MAX_EPS}mass.png")


def main(args):
    if args.train:
        start = time.time()
        ppo_rews, ppo_mass = train_ppo()
        with open(f"results/ppo{MAX_EPS}rews.pkl", "wb") as file:
            pickle.dump(ppo_rews, file)
        with open(f"results/ppo{MAX_EPS}mass.pkl", "wb") as file:
            pickle.dump(ppo_mass, file)
        dqn_rews, dqn_mass = train_dqn()
        with open(f"results/dqn{MAX_EPS}rews.pkl", "wb") as file:
            pickle.dump(dqn_rews, file)
        with open(f"results/dqn{MAX_EPS}mass.pkl", "wb") as file:
            pickle.dump(dqn_mass, file)
        end = time.time()
        total_time = end - start
        print(
            f"Fininished training {MAX_EPS} episodes in {total_time // 3600} hours {(total_time % 3600) // 60} minutes {(total_time % 3600) % 60} seconds"
        )

    if args.plot:
        with open(f"results/ppo{MAX_EPS}rews.pkl", "rb") as file:
            ppo_rews = pickle.load(file)
        with open(f"results/ppo{MAX_EPS}mass.pkl", "rb") as file:
            ppo_mass = pickle.load(file)
        with open(f"results/dqn{MAX_EPS}rews.pkl", "rb") as file:
            dqn_rews = pickle.load(file)
        with open(f"results/dqn{MAX_EPS}mass.pkl", "rb") as file:
            dqn_mass = pickle.load(file)
        make_plot(ppo_rews, ppo_mass, dqn_rews, dqn_mass)
    if args.show:
        show_models()
    if args.tune:
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction="maximize", storage=storage)
        study.optimize(tune_PPO, n_trials=30)
        run_server(storage)


def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Train roadAI MARL.")
    parser.add_argument(
        "-s", "--show", action="store_true", help="Shows pretrained model"
    )
    parser.add_argument("-t", "--train", action="store_true", help="Trains model")
    parser.add_argument("-m", "--plot", action="store_true", help="Makes plot")
    parser.add_argument("-o", "--tune", action="store_true", help="Tunes with Optuna")

    return parser.parse_args()


if __name__ == "__main__":
    # study = optuna.create_study()
    # study.optimize(objective, n_trials = 10)

    # study.best_params
    args = parse_args()
    print(args)
    main(args)
