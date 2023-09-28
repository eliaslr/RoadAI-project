import numpy as np
import ray 
from ray.rllib.algorithms import ppo


class TruckAgent:
    def __init__(self, id, pos_y, pos_x, ground, env, view_height):
        self.id = id
        self.pos_x = pos_x
        self.pos_y = pos_y
        # We have to keep track of whats under the truck when we replace tiles on the map
        self._ground = ground

        # We need to know if its filled or not to know what direction its "supposed" to go
        self.filled = False
        # Remember previous state of the agent to calculate reward later
        self.prev_agent = None
        self.capacity = 1000
        self.env = env
        self.collided = False
        self.view_height = view_height

    # This is a 2d raycasting where phi = pi / 4
    # The viewcone dims could be changed but you would then we to change 
    # the input dims for the algos
    def view_cone(self):
        cone = []
        for i in range(self.obs_space):
            for j in range(2*i + 1):
                if self._in_bounds:
                    cone.append((selfpos_y + i, self.pos_x - i + j, env.map[self.pos_y + i, self._pos_x + j))
        return cone

    def _in_bounds(self, pos, map):
        return (0 < pos[0] < map.shape[0]) and (0 < pos[1] < map.shape[1])

    def step(self, map):
        self.prev_agent = {
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "ground": self._ground,
            "filled": self.filled,
            "collided": self.collided,
        }

        # For now we just move randomly
        if dx or dy:
            map[self.pos_y, self.pos_x] = self._ground
            if 0 < self.pos_x + dx < map.shape[1]:
                self.pos_x += dx
            if 0 < self.pos_y + dy < map.shape[0]:
                self.pos_y += dy
            if map[self.pos_y, self.pos_x] == -1:
                self.collided = True
            else:
                self.collided = False
                self._ground = map[self.pos_y, self.pos_x]
                map[self.pos_y, self.pos_x] = -1

        # fill if were by an excavators
        adj = [
            (self.pos_y + 1, self.pos_x),
            (self.pos_y - 1, self.pos_x),
            (self.pos_y, self.pos_x + 1),
            (self.pos_y, self.pos_x - 1),
        ]

        for pos in adj:
            if not self._in_bounds(pos, map):
                continue
            if map[pos[0], pos[1]] == -3:
                self.filled = True
                break
            elif map[pos[0], pos[1]] == -2 and self.filled:
                self.filled = False
                self.holes[pos] -= self.capacity
                if self.holes[pos] <= 0:
                    map[pos[0], pos[1]] = 1
