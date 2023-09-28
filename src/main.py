from enviroment import RoadEnv
import reward
import matplotlib.pyplot as plt
# import hydra


def main():
    #ray.init()
    env = RoadEnv(reward.reward)
    rewards = []
    NUM_OF_TRAINING_EPS = 10
    #Train for X eps
    for _ in range(NUM_OF_TRAINING_EPS):
         rewards.append(env.eval_episode(render_mode = "console", train=True))
    # Show an episode to see how the system performs
    env.eval_episode(render_mode = "pygame", train=False)

    # Show Metrics
    plt.title("Average Rewards per episode")
    plt.plot(range(1, NUM_OF_TRAINING_EPS), rewards)
    plt.show()


if __name__ == "__main__":
    main()
