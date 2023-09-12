from enviroment import RoadEnv
import reward
import hydra


def main():
    env = RoadEnv(reward.reward)
    env.eval_episode()
    

if __name__ == "__main__":
    main()
