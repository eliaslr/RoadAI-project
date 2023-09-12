import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv

class TruckAgent:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def step(self, obs_space):
        # For now we just move randomly
        move = np.random.randint(0,4)
        #Todo add bounds checking
        if move == 0:
            self.pos_x +=1
        elif move == 1:
            self.pos_x -=1
        elif move == 2:
            self.pos_y +=1
        else:
            self.pos_y -=1


class RoadEnv(AECEnv):
    def __init__(self, reward_func):
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
            agent.step(None)
            rewards.append(None)
            #agent.update()
        return None, True

    
    def render(self, render_mode="console"):
        (H,W) = self.map.shape
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
        #TODO add pygame rendering 
        elif render_mode == "pygame":
            pass



    def eval_episode(self):
        self.map = self.generate_map()
        self.curr_step = 0
        self.curr_ep += 1
        term = False
        while not term:
            self.curr_step += 1
            rewards, term = self.step()
            self.render()

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
            start_pos = np.random.randint(0, H), np.random.randint(0, W)
            self.agents.append(TruckAgent(start_pos[0], start_pos[1]))
            map[start_pos[0], start_pos[1]] = -1
        
        #TODO add topological features / noise

        # For now we assume 20% of the road is to be filled
        num_holes = H * W // 5
        mass_per_hole = 1000 #Number of kilograms of road mass per square
        unfinished_road_dim = np.random.randint(0, W)
        curr_hole_pos = [H, W]
        for i in range(num_holes):
            if i % unfinished_road_dim == 0:
                curr_hole_pos[0] -= 1
                curr_hole_pos[1] = W - 1
            map[curr_hole_pos[0], curr_hole_pos[1]] = -2
            self.holes.append((mass_per_hole, curr_hole_pos))
            curr_hole_pos[1] -= 1
        return map
            
