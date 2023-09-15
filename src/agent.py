class TruckAgent:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def step(self, act_space, obs_space):
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


