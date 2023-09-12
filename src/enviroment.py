import gymnasium as gym
import numpy as np
from pettingzoo import AEC

class RoadEnv(AEC):
    def __init __(self):
        self._action_spaces = {}
        self._observation_spaces = {}
    
    def generate_map(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        
        
