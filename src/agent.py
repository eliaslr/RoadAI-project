import numpy as np


class TruckAgent:
    def __init__(self, pos_y, pos_x, ground, holes, agent_num):
        self.pos_x = pos_x
        self.pos_y = pos_y
        # We have to keep track of whats under the truck when we replace tiles on the map
        self._ground = ground

        # We need to know if its filled or not to know what direction its "supposed" to go
        self.filled = False
        # Remember previous state of the agent to calculate reward later
        self.prev_agent = None
        self.capacity = 1000
        self.holes = holes
        self.collided = False
        self.agent_num = agent_num

    def _in_bounds(self, pos, env):
        return (0 < pos[0] < env.map.shape[0]) and (0 < pos[1] < env.map.shape[1])

    def step(self, map, act_space, env):
        self.prev_agent = {
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "ground": self._ground,
            "filled": self.filled,
            "collided": self.collided,
        }

        # For now we just move randomly
        dy = act_space[0].sample()
        dx = act_space[1].sample()

        if dx or dy:
            map[self.pos_y, self.pos_x] = self._ground
            if 0 < self.pos_x + dx < map.shape[1]:
                self.pos_x += dx
            if 0 < self.pos_y + dy < map.shape[0]:
                self.pos_y += dy
            if map[self.pos_y, self.pos_x] <= -3 or map[self.pos_y, self.pos_x] == -1:
                self.collided = True
            else:
                self.collided = False
                self._ground = map[self.pos_y, self.pos_x]
                map[self.pos_y, self.pos_x] = -3-self.agent_num

        # fill if were by an
        adj = [
            (self.pos_y + 1, self.pos_x),
            (self.pos_y - 1, self.pos_x),
            (self.pos_y, self.pos_x + 1),
            (self.pos_y, self.pos_x - 1),
        ]

        for pos in adj:
            if not self._in_bounds(pos, env):
                continue
            if map[pos[0], pos[1]] == -1:
                self.filled = True

            elif map[pos[0], pos[1]] == -2 and self.filled:
                self.filled = False
                self.holes[pos] -= self.capacity
                if self.holes[pos] <= 0:
                    map[pos[0], pos[1]] = 1


    def deep_step(self, env, action):

        self.prev_agent = {
                            "pos_x"     : self.pos_x,
                            "pos_y"     : self.pos_y,
                            "ground"    : self._ground,
                            "filled"    : self.filled,
                            }


        dx = 0
        dy = 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1


        if dx or dy:
            env.map[self.pos_y, self.pos_x] = self._ground
            if 0 < self.pos_x + dx < env.map.shape[1]:
                self.pos_x += dx
            if 0 < self.pos_y + dy < env.map.shape[0]:
                self.pos_y += dy

            if env.map[self.pos_y, self.pos_x] == -1 or env.map[self.pos_y, self.pos_x] <= -3:
                self.collided = True
            else:
                self.collided = False
                self._ground = env.map[self.pos_y, self.pos_x]
                env.map[self.pos_y, self.pos_x] = -3 -self.agent_num
            self._ground = env.map[self.pos_y, self.pos_x]
            env.map[self.pos_y, self.pos_x] = -3 -self.agent_num

        #fill or empty
        adj = [
            (self.pos_y + 1, self.pos_x),
            (self.pos_y - 1, self.pos_x),
            (self.pos_y, self.pos_x + 1),
            (self.pos_y, self.pos_x - 1),
        ]

        for pos in adj:
            if not self._in_bounds(pos, env):
                continue
            if env.map[pos[0], pos[1]] == -1:
                self.filled = True

            elif env.map[pos[0], pos[1]] == -2 and self.filled:
                self.filled = False
                self.holes[pos] -= self.capacity
                if self.holes[pos] <= 0:
                    map[pos[0], pos[1]] = 1
