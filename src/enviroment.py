import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv

class TruckAgent:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def step(self, obs_space):
        # For now we just move randomly


class RoadEnv(AECEnv):
    def __init __(self, reward_func):
        self._action_spaces = {}
        self._observation_spaces = {}
        self.num_agents = 0
        self.agents = []
        self.holes = []
        self.reward_func = reward_func
        self.avg_CO2 = 0

    #Based on the example given in custom env tutorial from petting zoo
    #Returns rewards and termination status
    def step(self):
        for agent in self.agents:
            agent.step(self.map)
            agent.update()
    
    def render(self, render_mode="console"):
        (H,W) = map.shape
        if render_mode == "console":
            for i in range(H):
                for j in range(W):
                    print(map[i,j])
        #TODO add pygame rendering 
        elif render_mode == "pygame":
            pass



    def eval_episode(self):
        map = generate_map()


    def generate_map(self, seed=None, min_h = 10, min_w = 10, max_h = 1000, max_w = 1000):
        if seed != None:
            np.random.seed(seed)
        
        H = np.random.randint(min_h, max_h)
        W = np.random.randint(min_w, max_w)


        #For now we have a flat map 
        map = np.ones((H, W))
        self.num_agents = np.random.randint(1,10)
        for _ in range(self.num_agents):
            start_pos = np.random.randint(0, H), np.random.randint(0, W)
            self.agents.append(TruckAgent(start_pos[0], start_pos[1]))
            map[start_pos[0], start_pos[1]] = -1
        
        # For now we assume 20% of the road is to be filled
        num_holes = H * W // 5
        mass_per_hole = #Number of kilograms of road mass per square
        for i in range(num_holes):
            self.holes.append()
        return map
            
