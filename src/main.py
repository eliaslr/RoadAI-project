from enviroment import RoadEnv
from ppo import PPO

import reward
import matplotlib.pyplot as plt

# import hydra


def main():
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

    # Show Metrics
    plt.title("Average Rewards per episode")
    plt.plot(range(NUM_OF_TRAINING_EPS), rewards)
    plt.show()


if __name__ == "__main__":
    main()
