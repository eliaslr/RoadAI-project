import numpy as np
from pettingzoo import ParallelEnv
import gymnasium as gym
import pygame
from agent import TruckAgent

WINDOW_H = 600
WINDOW_W = 600


class RoadEnv(ParallelEnv):
    def __init__(self, reward_func, max_agents = None):
        self._action_spaces = {}
        self._observation_spaces = {}
        self.agents = []
        self.holes = []
        self.curr_ep = 0
        self.reward_func = reward_func

    #Based on the example given in custom env tutorial from petting zoo
    #Returns rewards and termination status
    def step(self):
        rewards = []
        for agent in self.agents:
        #TODO update action/obs space for each agent
            #Cardinal movement for each truck
            self._action_spaces[agent] = gym.spaces.Tuple((gym.spaces.Discrete(3, start = agent.pos_y-1),
                                                gym.spaces.Discrete(3, start = agent.pos_x-1)))
            agent.step(None, None)
            #reward = self.reward_func(agent, self.map)
            #agent.update(reward)
        return None, True

    
    def render(self, render_mode="console"):
        (H, W) = self.map.shape
        if render_mode == "console":
            print(f"Episode: {self.curr_ep}, Step: {self.curr_step}")
            print("-" *W)
            for i in range(H):
                for j in range(W):
                    if self.map[i,j] == -1:
                        print("X", end="")
                    elif self.map[i,j] == -2:
                        print("U", end="")
                    elif self.map[i,j] == -3:
                        print("E", end="")
                    elif self.map[i,j] < 10:
                        print(self.map[i,j], end="")
                    else:
                        print("#", end="")
                print("")
        #TODO add pygame rendering for sprites
        elif render_mode == "pygame":
            for i in range(H):
                for j in range(W):
                    pos = pygame.Rect((self._margin + j * self._s_size, 
                                       self._margin + i * self._s_size), (self._s_size, self._s_size))
                    if self.map[i,j] == -1:
                        pygame.draw.rect(self._screen, (255, 0, 0), pos)
                    elif self.map[i,j] == -2:
                        pygame.draw.rect(self._screen, (0,0,255), pos)
                    elif self.map[i,j] == -3:
                        pygame.draw.rect(self._screen, (0, 158, 158), pos)
                    else:
                        c = min(200, self.map[i,j])
                        pygame.draw.rect(self._screen, (200 - c, 200 - c, 200 - c), pos)
            pygame.display.flip()
        elif render_mode == None:
            return

    def reset(self):
        pass
    


    def eval_episode(self, render_mode = "console"):
        self.generate_map()
        (H,W) = self.map.shape
        if render_mode == "pygame":
            pygame.init()
            # TODO ADD CONSTANTS IN HYDRA
            self._margin = 50 
            self._s_size = (WINDOW_H - self._margin) // H if H > W else (WINDOW_W - self._margin) // W
            self._screen = pygame.display.set_mode((self._s_size * W + 2 * self._margin, 
                                                    self._s_size * H + 2 * self._margin))
            self._screen.fill((0, 0, 0), rect= None)
        self.curr_step = 0
        self.curr_ep += 1
        term = False
        while not term:
            self.curr_step += 1
            #rewards, term = self.step()
            # TODO avg rewards/ call back to update agents
            self.render(render_mode = render_mode)
            if render_mode == "pygame":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        term = True 

    
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
                    self.map[start_pos[0] + (i - 2 * w) - h, start_pos[1] + w] += rand_val
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
        
    def generate_map(self, seed=None, min_h = 50, min_w = 50, max_h = 100, max_w = 100):
        if seed != None:
            np.random.seed(seed)
        
        H = np.random.randint(min_h, max_h)
        W = np.random.randint(min_w, max_w)

        #Add terrain noise
        self.map = np.random.randint(10, size = (H, W))
        self._num_agents = np.random.randint(3,10)
        for _ in range(self._num_agents):
        #TODO add option for fixed startpositions/A deposit where the material is hauled from
            start_pos = (np.random.randint(0, H), np.random.randint(0, W))
            self.agents.append(TruckAgent(start_pos[0], start_pos[1]))
            self.map[start_pos[0], start_pos[1]] = -1
        
        #TODO add topological features / noise
        num_of_features = np.random.randint(3, 10)
        for _ in range(num_of_features):
            start_pos = (np.random.randint(0,  4 * H // 5 - 20), np.random.randint(0, W - 20))
            size = (np.random.randint(5, max(20, H - start_pos[0])), 
                    np.random.randint(5, max(20, W - start_pos[1])))
            mag = 3
            self._topograph_feature(start_pos, size[0], size[1], mag)


        # For now we assume 20% of the road is to be filled
        num_holes = H * W // 5
        num_excavators = 3
        for _ in range(num_excavators):
            #Excavators always start at the top of the map
            pos = (np.random.randint(0, H // 2), np.random.randint(0, W))
            self.map[pos[0], pos[1]] = -3


        mass_per_hole = 1000 #Number of kilograms of road mass per square
        for i in range(4 * H // 5, H):
            for j in range(W):
                self.map[i, j] = -2
                #holes.append((mass_per_hole, curr_hole_pos))

        # Unsure if max_dim is needed
        # TODO Add map gen from drone data
        # 1. Run segmentation and classification model on image
        # 2. Deskew/align the image
        # 3. Translate the output classes to map values
        # 4. Add some noise to the map values to better simulate real road conditions
        def gen_map_from_image(self, image_file, classification_model):
            pass     
            
