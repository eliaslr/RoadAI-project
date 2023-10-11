from enviroment import RoadEnv
from ppo import PPO

import reward
import matplotlib.pyplot as plt

# import hydra


def main():
    # ray.init()
    env = RoadEnv(reward.reward)
    ppo = PPO(env, 0.005, 0.5)
    rewards = []
    NUM_OF_TRAINING_EPS = 5
    # Train for X eps
    for _ in range(NUM_OF_TRAINING_EPS):
        env.reset()
        ppo.train(render_mode="pygame")
        # rewards.append(env.eval_episode(render_mode="pygame", train=True))
    # Show an episode to see how the system performs

    # Show Metrics
    # plt.title("Average Rewards per episode")
    # plt.plot(range(NUM_OF_TRAINING_EPS), rewards)
    # plt.show()


if __name__ == "__main__":
    main()
