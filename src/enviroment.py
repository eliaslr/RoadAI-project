import numpy as np
from pettingzoo import ParallelEnv
import gymnasium as gym
import pygame
from agent import TruckAgent

WINDOW_H = 600
WINDOW_W = 600


# Note that we use (y,x) instead of (x, y) in our coordinates
class RoadEnv(ParallelEnv):
    def __init__(self, reward_func, max_agents = None, n_agents = None):
        self._action_spaces = {}
        # self._observation_spaces = {} # The obs space is the entire map
        self.agents = []
        self.holes = {}
        self.n_agents = n_agents
        self.curr_ep = 0
        self.reward_func = reward_func
        self.excavators = []
        self.curr_step = 0
        self._screen = None
    # Based on the example given in custom env tutorial from petting zoo
    # Returns rewards and termination status
    def step(self):
        rewards = []
        for agent in self.agents:
            # TODO update action/obs space for each agent
            # Cardinal movement for each truck
            self._action_spaces[agent] = gym.spaces.Tuple(
                (gym.spaces.Discrete(3, start=-1), gym.spaces.Discrete(3, start=-1))
            )
            agent.step(self.map, self._action_spaces[agent], self)
            reward = self.reward_func(agent, self)

            # agent.update(reward)
            # Add metrics of avg reward
        return None, False

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
            if not self._screen:
                pygame.init()
                # TODO ADD CONSTANTS IN HYDRA
                self._margin = 50
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
                    if self.map[i, j] == -1:
                        pygame.draw.rect(self._screen, (0, 158, 158), pos)
                    elif self.map[i, j] == -2:
                        pygame.draw.rect(self._screen, (0, 0, 255), pos)
                    elif self.map[i, j] <= -3:
                        pygame.draw.rect(self._screen, (255, 0, 0), pos)
                    else:
                        c = min(200, self.map[i, j])
                        pygame.draw.rect(self._screen, (200 - c, 200 - c, 200 - c), pos)
            pygame.display.flip()
            pygame.time.delay(50)
        elif render_mode is None:
            return

    def reset(self, seed=None, min_h=50, min_w=50, max_h=100, max_w=100, n_agents = None):
        self._action_spaces = {}
        self.agents = []
        self.holes = {}
        self.curr_ep = 0
        # self.reward_func = reward_func
        self.excavators = []

        self.generate_map(seed = seed, min_h = min_h, max_h = max_h, min_w = min_w, max_w = max_w)

        return self.map

    # Evaluates one episode of play
    def eval_episode(self, render_mode="console"):
        self.generate_map()
        (H, W) = self.map.shape
        if render_mode == "pygame":
            pygame.init()
            # TODO ADD CONSTANTS IN HYDRA
            self._margin = 50
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
        term = False
        while not term:
            self.curr_step += 1
            rewards, term = self.step()
            # TODO avg rewards/ call back to update agents
            self.render(render_mode=render_mode)
            if render_mode == "pygame":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        term = True

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
        if seed != None:
            np.random.seed(seed)

        if min_h == max_h:
            H = min_h
        else:
            H = np.random.randint(min_h, max_h)

        if min_w == max_w:
            W = min_w
        else:
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

        if self.n_agents:
            self._num_agents = self.n_agents
        else:
            self._num_agents = np.random.randint(3, 10)
        for i in range(self._num_agents):
            # TODO add option for fixed startpositions/A deposit where the material is hauled from
            start_pos = (np.random.randint(0, 4 * H // 5), np.random.randint(0, W))
            # Make sure we start off in a flat space
            while self.map[start_pos[0], start_pos[1]] >= 10:
                start_pos = (np.random.randint(0, 4 * H // 5), np.random.randint(0, W))
            self.agents.append(
                TruckAgent(
                    start_pos[0],
                    start_pos[1],
                    self.map[start_pos[0], start_pos[1]],
                    self.holes,
                    i
                )
            )
            # to make the agents distinguishable for the network
            self.map[start_pos[0], start_pos[1]] = -3-i
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


    def step_deep(self, actions, render):
        print(actions)
        reward = 0
        for i in range(len(self.agents)):
            self.agents[i].deep_step(self, actions[i])
            reward += self.reward_func(self.agents[i], self)
        self.curr_step += 1
        if render:
            self.render(render_mode = render)

        return self.map.flatten(), reward/len(self.agents), False, False
