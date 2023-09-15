from enviroment import RoadEnv
import reward
import ray
import hydra


def main(): -> None
    ray.init()
    env = RoadEnv(reward.reward)
    for _ in range(1):
        env.eval_episode(render_mode = "console")
    

if __name__ == "__main__":
    main()
