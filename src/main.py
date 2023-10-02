from omegaconf import DictConfig
from enviroment import RoadEnv
import reward
#import ray
import hydra

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    #ray.init()
    env = RoadEnv(reward.reward, cfg)
    for _ in range(cfg.N_EPISODES):
        env.eval_episode(render_mode = "pygame")


if __name__ == "__main__":
    main()
