import numpy as np

class TruckAgent:
    def __init__(self, pos_y, pos_x, ground):
        self.pos_x = pos_x
        self.pos_y = pos_y
        #We have to keep track of whats under the truck when we replace tiles on the map
        self._ground = ground

        #We need to know if its filled or not to know what direction its "supposed" to go
        self.filled = False
        # Remember previous state of the agent to calculate reward later
        self.prev_agent = None

    def step(self, obs_space, act_space):

        self.prev_agent = {
                            "pos_x"     : self.pos_x,
                            "pos_y"     : self.pos_y,
                            "ground"    : self._ground,
                            "filled"    : self.filled,
                            }

        # For now we just move randomly
        dy = act_space[0].sample()
        dx = act_space[1].sample()

        if dx or dy:
            obs_space[self.pos_y, self.pos_x] = self._ground
            if 0 < self.pos_x + dx < obs_space.shape[1]:
                self.pos_x += dx
            if 0 < self.pos_y + dy < obs_space.shape[0]:
                self.pos_y += dy
            self._ground = obs_space[self.pos_y, self.pos_x]
            obs_space[self.pos_y, self.pos_x] = -1
