import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from agent import TruckAgent

WINDOW_H = 600
WINDOW_W = 600
MAX_STEPS = 5000  # Episode length


# Note that we use (y,x) instead of (x, y) in our coordinates
class RoadEnv(gym.Env):
    def __init__(self, reward_func, max_agents=None, render_mode=None):
        self.reward_func = reward_func
        self.view_dist = 15  # Parameter for how far each truck can see
        self.curr_ep = -1
        self.avg_rewards = []  # Avg rewards for every episode
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                "filled": spaces.Discrete(2),
                "pos": spaces.Box(0, 1000, shape=(2,), dtype=int),
                "adj": spaces.Box(-10, 1000, shape=(4,), dtype=int),
                "target": spaces.Box(0, 1000, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(5)
        if render_mode == "pygame":
            pygame.init()
            # TODO ADD CONSTANTS IN HYDRA
            self._margin = 25
        self.reset()

    def _reset_screen(self):
        (H, W) = self.map.shape
        self._s_size = (
            (WINDOW_H - self._margin) // H if H > W else (WINDOW_W - self._margin) // W
        )
        self._screen = pygame.display.set_mode(
            (
                self._s_size * W + 2 * self._margin,
                self._s_size * H + 2 * self._margin,
            )
        )

    # Updates agents states, reward, observations
    # Called from learning algorithm

    def step(self, action):
        self.curr_step += 1
        agent = self.agents[self.curr_step % self._num_agents]
        agent.step(action)
        obs = agent.observe()
        # Get reward
        rew = self.reward_func(agent, self)
        self.avg_reward = (
            self.avg_reward * (self.curr_step - 1) + rew
        ) / self.curr_step
        term = False
        if self.curr_step > MAX_STEPS:
            term = True
            self.avg_rewards.append(self.avg_reward)
        self.render()
        return (
            obs,
            rew,
            term,
            False,
            {},
        )

    # Renders the environment
    # Accepts 3 modes:
    # Console prints the enviroment in ascii to console
    # Pygame renders a graphical view
    # None skips rendering
    def render(self):
        (H, W) = self.map.shape
        if self.render_mode == "console":
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
        elif self.render_mode == "pygame":
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
                        if self.agents[self.map[i, j] * -1 - 3].out_of_bounds:
                            pygame.draw.rect(self._screen, (0, 255, 255), pos)
                        elif self.agents[self.map[i, j] * -1 - 3].filled:
                            pygame.draw.rect(self._screen, (0, 200, 150), pos)
                        else:
                            pygame.draw.rect(self._screen, (255, 0, 0), pos)
                    elif self.map[i, j] == -2:
                        pygame.draw.rect(self._screen, (0, 0, 255), pos)
                    elif self.map[i, j] == -1:
                        pygame.draw.rect(self._screen, (0, 158, 158), pos)
                    else:
                        c = min(200, self.map[i, j])
                        pygame.draw.rect(self._screen, (200 - c, 200 - c, 200 - c), pos)
            pygame.display.flip()
            # pygame.time.delay(25)
        elif self.render_mode is None:
            return

        # Resets the env and generates new map

    # Should be called inbetween episodes
    def reset(self, seed=None):
        self.agents = []
        self.holes = {}
        self.excavators = []
        self.avg_reward = 0
        self.generate_map(seed=seed)
        if self.render_mode:
            self._reset_screen()
        self.curr_step = 0
        self.curr_ep += 1
        obs, _, _, _, _ = self.step(0)
        return obs, {}

    # Generates topographic "hills" where the hills get larger as you get to the center
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

    # Generates the grid and populates it with agents
    def generate_map(self, seed=None, min_h=50, min_w=50, max_h=75, max_w=75):
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
            mag = 2
            self._topograph_feature(start_pos, size[0], size[1], mag)

        num_excavators = 10
        for _ in range(num_excavators):
            # Excavators always start at the top of the map
            pos = (np.random.randint(0, H // 2), np.random.randint(0, W))
            # We need this for reward function
            self.excavators.append(pos)
            # Make sure we start off in a flat space
            while self.map[pos[0], pos[1]] >= 10:
                pos = (
                    np.random.randint(H // 4, H // 2),
                    np.random.randint(W // 4, 3 * W // 4),
                )
            self.map[pos[0], pos[1]] = -1

        self._num_agents = np.random.randint(3, 10)
        for i in range(self._num_agents):
            # TODO add option for fixed startpositions/A deposit where the material is hauled from
            start_pos = (np.random.randint(1, 4 * H // 5), np.random.randint(1, W))
            # Make sure we start off in a flat space
            while self.map[start_pos[0], start_pos[1]] >= 10:
                start_pos = (np.random.randint(1, 4 * H // 5), np.random.randint(1, W))
            self.agents.append(
                TruckAgent(
                    i,
                    start_pos[0],
                    start_pos[1],
                    self.map[start_pos[0], start_pos[1]],
                    self,
                    self.view_dist,
                )
            )
            self.map[start_pos[0], start_pos[1]] = -i - 3

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
