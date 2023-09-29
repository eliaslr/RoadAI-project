import numpy as np
import pygame
import gymnasium as gym
from ppo import PPO
from gymnasium import spaces
from agent import TruckAgent

WINDOW_H = 600
WINDOW_W = 600
MAX_STEPS = 100000


# Note that we use (y,x) instead of (x, y) in our coordinates
class RoadEnv(gym.Env):
    def __init__(self, reward_func, max_agents=None):
        self.action_space = {}
        self.agents = []
        self.observation_space = {}
        self.holes = {}
        self.curr_ep = 0
        self.reward_func = reward_func
        self.avg_rewards = {}
        self.excavators = []
        # View distance of trucks
        self.view_dist = 4
        self.algo = PPO(self, 0.005, 0.2)


    # Based on the example given in custom env tutorial from petting zoo
    # Returns rewards and termination status
    def step(self):
        agent_rewards = [0] * self._num_agents
        actions = [0] * self._num_agents
        #term_status = {}
        #infos = {{} for _ in range(len(self.agents))} # Only for ray api req
        #truncs = {{} for _ in range(len(self.agents))} # Only for ray api req
        for agent in self.agents:
            # TODO update action/obs space for each agent
            # Cardinal movement for each truck
            self.action_space[agent.id] = spaces.MultiDiscrete([agent.pos_y + 1, agent.pos_x + 1],
                                                                start=[agent.pos_y -1, agent.pos_x +1])
            self.observation_space[agent.id] = agent.view()
            # Get next action
            action = self.algo.action(self.observation_space[agent.id], agent)
            agent.step(action)
            # Get reward
            reward = self.reward_func(agent, self)
            agent_rewards[agent.id] = reward
            actions[agent.id] = action
            #term_status[agent] = False # Change this to custom
            # Running avg
            self.avg_rewards[agent] += (
                reward - self.avg_rewards[agent]
            ) / self.curr_step
        return (self.observation_space, actions, agent_rewards)

    # Renders the environment accepts 3 modes
    # Console prints the enviroment in ascii to console
    # Pygame renders a graphical view
    # None skips rendering
    def render(self, render_mode="console"):
        (H, W) = self.map.shape
        if render_mode == "console":
            print(f"Episode: {self.curr_ep}, Step: {self.curr_step}")
            print("-" * W)
            for i in range(H):
                for j in range(W):
                    if self.map[i, j] == -1:
                        print("X", end="")
                    elif self.map[i, j] == -2:
                        print("U", end="")
                    elif self.map[i, j] == -3:
                        print("E", end="")
                    elif self.map[i, j] < 10:
                        print(self.map[i, j], end="")
                    else:
                        print("#", end="")
                print("")
        elif render_mode == "pygame":
            self._screen.fill((0, 0, 0), rect=None)
            for i in range(H):
                for j in range(W):
                    pos = pygame.Rect(
                        (
                            self._margin + j * self._s_size,
                            self._margin + i * self._s_size,
                        ),
                        (self._s_size, self._s_size),
                    )
                    if self.map[i, j] <= -3:
                        pygame.draw.rect(self._screen, (255, 0, 0), pos)
                        # TODO add rendering for view cones
                        #cone = self.observation_space[self.map[i, j] * -1 - 3]
                        #for sq in cone:
                        #    pos = pygame.Rect((self._margin + sq[1] * self._s_size,
                        #                      self._margin + sq[0] * self._s_size),
                        #                      (self._s_size, self._s_size))
                        #    pygame.draw.rect(self._screen, (255, 255, 255, 1), pos)
                    elif self.map[i, j] == -2:
                        pygame.draw.rect(self._screen, (0, 0, 255), pos)
                    elif self.map[i, j] == -1:
                        pygame.draw.rect(self._screen, (0, 158, 158), pos)
                    else:
                        c = min(200, self.map[i, j])
                        pygame.draw.rect(self._screen, (200 - c, 200 - c, 200 - c), pos)
            pygame.display.flip()
            pygame.time.delay(50)
        elif render_mode is None:
            return

    def reset(self):
        self.action_space = {}
        self.observation_space = {}
        self.agents = []
        self.holes = {}
        self.curr_ep = 0
        # self.reward_func = reward_func
        self.excavators = []

        self.generate_map()

        return self.observation_space

    # Evaluates one episode of play
    def eval_episode(self, train=True, render_mode="console"):
        self.generate_map()
        self.avg_rewards = {}
        for agent in self.agents:
            self.avg_rewards[agent] = 0
        (H, W) = self.map.shape
        if render_mode == "pygame":
            pygame.init()
            # TODO ADD CONSTANTS IN HYDRA
            self._margin = 25
            self._s_size = (
                (WINDOW_H - self._margin) // H
                if H > W
                else (WINDOW_W - self._margin) // W
            )
            self._screen = pygame.display.set_mode(
                (
                    self._s_size * W + 2 * self._margin,
                    self._s_size * H + 2 * self._margin,
                )
            )
        self.curr_step = 0
        self.curr_ep += 1
        # Main evaluation loop
        term = False
        while not term:
            self.curr_step += 1
            if self.curr_step > MAX_STEPS:
                term = True
            if train:
                self.algo.learn()
            else:
                self.step()
            if self.curr_step % 100 == 0:
                print(self.curr_step)
                print(np.mean(list(self.avg_rewards.values())))
            self.render(render_mode=render_mode)
            if render_mode == "pygame":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        term = True
        return np.mean(self.avg_rewards.values())

    # Generates
    def _topograph_feature(self, start_pos, h, w, mag):
        last_val = 1
        while w > 0 and h > 0:
            rand_val = np.random.randint(last_val, last_val * mag)
            for i in range(2 * w + 2 * h):
                if i < w:
                    self.map[start_pos[0], start_pos[1] + i] += rand_val
                elif i <= 2 * w:
                    self.map[start_pos[0] + h, start_pos[1] + i - w] += rand_val
                elif i < 2 * w + h:
                    self.map[start_pos[0] + i - 2 * w, start_pos[1]] += rand_val
                else:
                    self.map[
                        start_pos[0] + (i - 2 * w) - h, start_pos[1] + w
                    ] += rand_val
            w -= 2
            h -= 2
            start_pos = (start_pos[0] + 1, start_pos[1] + 1)
            last_val = rand_val
        rand_val = np.random.randint(last_val, last_val * mag)
        while w >= 0:
            self.map[start_pos[0], start_pos[1] + w] += rand_val
            w -= 1
        while h >= 0:
            self.map[start_pos[0] + h, start_pos[1]] += rand_val
            h -= 1

    def generate_map(self, seed=None, min_h=50, min_w=50, max_h=100, max_w=100):
        if seed is not None:
            np.random.seed(seed)

        H = np.random.randint(min_h, max_h)
        W = np.random.randint(min_w, max_w)
        # Add terrain noise
        self.map = np.random.randint(10, size=(H, W))
        # TODO add topological features / noise
        num_of_features = np.random.randint(3, 10)
        for _ in range(num_of_features):
            start_pos = (
                np.random.randint(0, 4 * H // 5 - 20),
                np.random.randint(0, W - 20),
            )
            size = (
                np.random.randint(5, max(20, H - start_pos[0])),
                np.random.randint(5, max(20, W - start_pos[1])),
            )
            mag = 3
            self._topograph_feature(start_pos, size[0], size[1], mag)

        self._num_agents = np.random.randint(3, 10)
        for i in range(self._num_agents):
            # TODO add option for fixed startpositions/A deposit where the material is hauled from
            start_pos = (np.random.randint(0, 4 * H // 5), np.random.randint(0, W))
            # Make sure we start off in a flat space
            while self.map[start_pos[0], start_pos[1]] >= 10:
                start_pos = (np.random.randint(0, 4 * H // 5), np.random.randint(0, W))
            self.agents.append(
                TruckAgent(
                    i,
                    start_pos[0],
                    start_pos[1],
                    self.map[start_pos[0], start_pos[1]],
                    self,
                    self.view_dist
                )
            )
            self.map[start_pos[0], start_pos[1]] = -i - 3
        num_excavators = 3
        for _ in range(num_excavators):
            # Excavators always start at the top of the map
            pos = (np.random.randint(0, H // 2), np.random.randint(0, W))
            # Make sure we start off in a flat space

            """
                QUESTION
                should the excavators move frequently enough for that to happen during an episode or is it fine to just store their position?
            """
            # Assume that excavators are stationary
            # We need this for reward function
            self.excavators.append(pos)
            # Make sure we start off in a flat space
            while self.map[pos[0], pos[1]] >= 10:
                pos = (np.random.randint(0, H // 2), np.random.randint(0, W))
            self.map[pos[0], pos[1]] = -1

        mass_per_hole = 1000  # Number of kilograms of road mass per square
        for i in range(9 * H // 10, H):
            for j in range(W):
                self.map[i, j] = -2
                self.holes[(i, j)] = mass_per_hole

        # Unsure if max_dim is needed
        # TODO Add map gen from drone data
        # 1. Run segmentation and classification model on image
        # 2. Deskew/align the image
        # 3. Translate the output classes to map values
        # 4. Add some noise to the map values to better simulate real road conditions
        def gen_map_from_image(self, image_file, classification_model):
            pass
