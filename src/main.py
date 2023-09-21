from enviroment import RoadEnv
import reward
import ray


def main():
    #ray.init()
    env = RoadEnv(reward.reward)
    for _ in range(1):
        env.eval_episode(render_mode = "pygame")
    

if __name__ == "__main__":
    main()
