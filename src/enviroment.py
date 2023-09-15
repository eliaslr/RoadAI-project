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
            agent.step(None, None)
            #Cardinal movement for each truck
            self._action_spaces[agent] = gym.spaces.Tuple(gym.spaces.Discrete(3, start = agent.pos_y-1),
                                                gym.spaces.Discrete(3, start = agent.pos_x-1))
            reward = self.reward_func(agent, self.map)
            #agent.update(reward)
        return None, True

    
    def render(self, render_mode="console"):
        (H,W) = self.map.shape
        s_size = WINDOW_H // H if H > W else WINDOW_W // W
        margin = 50
        if render_mode == "console":
            print(f"Episode: {self.curr_ep}, Step: {self.curr_step}")
            print("-" *W)
            for i in range(H):
                for j in range(W):
                    if self.map[i,j] == -1:
                        print("X", end="")
                    elif self.map[i,j] == -2:
                        print("U", end="")
                    elif self.map[i,j] == 1:
                        print("o", end="")
                print("")
        #TODO add pygame rendering for sprites
        elif render_mode == "pygame":
            for i in range(H):
                for j in range(W):
                    pos = pygame.Rect((j * s_size,i * s_size), (s_size, s_size))
                    if self.map[i,j] == -1:
                        pygame.draw.rect(self._screen, (255, 0, 0), pos)
                    elif self.map[i,j] == -2:
                        pygame.draw.rect(self._screen, (0,0,255), pos)
                    else:
                        pygame.draw.rect(self._screen, (128, 128, 128), pos)
            pygame.display.flip()
        elif render_mode == None:
            return

    def reset(self):
        pass
    


    def eval_episode(self, render_mode = "console"):
        self.map = self.generate_map()
        if render_mode == "pygame":
            pygame.init()
            # TODO ADD CONSTANTS IN HYDRA
            self._screen = pygame.display.set_mode((WINDOW_H, WINDOW_W))
        self.curr_step = 0
        self.curr_ep += 1
        term = False
        while not term:
            self.curr_step += 1
            rewards, term = self.step()
            # TODO avg rewards/ call back to update agents
            self.render(render_mode = render_mode)
            if render_mode == "pygame":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        term = True 

    
    def _diamond_square(self, map, h):
        (H,W) = map.shape
        for i in range(1, H):
            rand_val = np.random.randint()
            #diamond step
            if i % 2 == 1:
                for j in range(i):
                    midpoint = (H // (2 * j), W // (2 * j))
                    map[midpoint[0], midpoint[1]] = 
            #square step
            else:
                for j in range(4 * (i-1)):
                    midpoint = ()
                    
        
    def generate_map(self, seed=None, min_h = 10, min_w = 10, max_h = 50, max_w = 50):
        if seed != None:
            np.random.seed(seed)
        
        H = np.random.randint(min_h, max_h)
        W = np.random.randint(min_w, max_w)


        #For now we have a flat map 
        map = np.ones((H, W))
        self._num_agents = np.random.randint(1,10)
        for _ in range(self._num_agents):
        #TODO add option for fixed startpositions/A deposit where the material is hauled from
            start_pos = (np.random.randint(0, H), np.random.randint(0, W))
            self.agents.append(TruckAgent(start_pos[0], start_pos[1]))
            map[start_pos[0], start_pos[1]] = -1
        
        #TODO add topological features / noise

        # For now we assume 20% of the road is to be filled
        num_holes = H * W // 5
        num_excavators = 1
        for _ in range(num_excavators):
            #Excavators always start at the top of the map
            pos = (np.random.randint(0, H // 2), np.random.randint(0, W))
            map[pos[0], pos[1]] = 100

        mass_per_hole = 1000 #Number of kilograms of road mass per square
        unfinished_road_dim = np.random.randint(1, W - 1)
        curr_hole_pos = [H, W]
        for i in range(num_holes):
            if i % unfinished_road_dim == 0:
                curr_hole_pos[0] -= 1
                curr_hole_pos[1] = W - 1
            map[curr_hole_pos[0], curr_hole_pos[1]] = -2
            self.holes.append((mass_per_hole, curr_hole_pos))
            curr_hole_pos[1] -= 1
        return map
            
        # Unsure if max_dim is needed
        # TODO Add map gen from drone data
        # 1. Run segmentation and classification model on image
        # 2. Deskew/align the image
        # 3. Translate the output classes to map values
        # 4. Add some noise to the map values to better simulate real road conditions
        def gen_map_from_image(self, image_file, classification_model):
            pass     
            
