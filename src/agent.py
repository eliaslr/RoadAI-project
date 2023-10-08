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
        self.out_of_bounds = False
        self.view_height = view_height
        self.dir = 1  # Start pointing North

    # This is a 2d raycasting where phi = pi / 4
    # The viewcone dims could be changed but you would then we to change
    # the input dims for the algos
    def view(self):
        cone = []
        if self.dir == 1:
            for i in range(self.view_height):
                for j in range(2*i + 1):
                    if self._in_bounds((self.pos_y - i, self.pos_x - i + j)):
                        cone.append((self.pos_y - i, self.pos_x - i + j,
                                     self.env.map[self.pos_y - i, self.pos_x - i + j]))
                    else:
                        cone.append((self.pos_y - i, self.pos_x - i + j, 1000))
        elif self.dir == 2:
            for i in range(self.view_height):
                for j in range(2*i + 1):
                    if self._in_bounds((self.pos_y + i, self.pos_x - i + j)):
                        cone.append((self.pos_y + i, self.pos_x - i + j,
                                     self.env.map[self.pos_y + i, self.pos_x - i + j]))
                    else:
                        cone.append((self.pos_y + i, self.pos_x - i + j, 1000))
        elif self.dir == 3:
            for i in range(self.view_height):
                for j in range(2*i + 1):
                    if self._in_bounds((self.pos_y - i + j, self.pos_x + i)):
                        cone.append((self.pos_y - i + j, self.pos_x + i,
                                     self.env.map[self.pos_y - i + j, self.pos_x + i]))
                    else:
                        cone.append((self.pos_y - i + j, self.pos_x + i, 1000))
        else:
            for i in range(self.view_height):
                for j in range(2*i + 1):
                    if self._in_bounds((self.pos_y - i + j, self.pos_x - i)):
                        cone.append((self.pos_y - i + j, self.pos_x - i,
                                     self.env.map[self.pos_y - i + j, self.pos_x - i]))
                    else:
                        cone.append((self.pos_y - i + j, self.pos_x - i, 1000))
        return cone

    def _in_bounds(self, pos):
        return (0 <= pos[0] < self.env.map.shape[0]) and (0 <= pos[1] < self.env.map.shape[1])

    def step(self, action):
        self.prev_agent = {
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "ground": self._ground,
            "filled": self.filled,
            "collided": self.collided,
        }
        # Change dir we are facing
        dx, dy = 0, 0
        if action == 1:
            dy = -1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1
        elif action == 4:
            dx = 1

        if dx or dy:
            self.dir = action
            self.env.map[self.pos_y, self.pos_x] = self._ground
            self.pos_x += dx
            self.pos_y += dy
            if not self._in_bounds((self.pos_y, self.pos_x)):
                self.out_of_bounds = True
                self.pos_x -= dx
                self.pos_y -= dy
            else:
                self.out_of_bounds = False

            if self.env.map[self.pos_y, self.pos_x] == -1 or self.env.map[self.pos_y, self.pos_x] <= -3:
                self.collided = True
            else:
                self.collided = False
                self._ground = self.env.map[self.pos_y, self.pos_x]
                self.env.map[self.pos_y, self.pos_x] = -self.id - 3

        # fill if were by an excavators
        adj = [
            (self.pos_y + 1, self.pos_x),
            (self.pos_y - 1, self.pos_x),
            (self.pos_y, self.pos_x + 1),
            (self.pos_y, self.pos_x - 1),
        ]

        for pos in adj:
            if not self._in_bounds(pos):
                continue
            if self.env.map[pos[0], pos[1]] == -3:
                self.filled = True
                break
            elif self.env.map[pos[0], pos[1]] == -2 and self.filled:
                self.filled = False
                self.env.holes[pos] -= self.capacity
                if self.env.holes[pos] <= 0:
                    self.env.map[pos[0], pos[1]] = 1
