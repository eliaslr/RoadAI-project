from omegaconf import DictConfig
from enviroment import RoadEnv
import reward
#import ray
import hydra

N_EPISODES = 1

# @hydra.main(config_path="conf", config_name="config")
def main() -> None:
    #ray.init()
    env = RoadEnv(reward.reward)
    for _ in range(N_EPISODES):
        env.eval_episode(render_mode = "pygame")


if __name__ == "__main__":
    main()
