import numpy as np


def reward(agent, env):
    tot_reward = 0
    previous = agent.prev_agent

    # Rewards
    idle_penalty = -0.5
    step_penalty = -0.1
    collision_pen = 0
    right_direction = 1
    out_of_bounds = -1

    filled_emptied = 1

    if agent.collided:
        tot_reward += collision_pen

    if agent.out_of_bounds:
        tot_reward += out_of_bounds

    if previous["pos_x"] == agent.pos_x and previous["pos_y"] == agent.pos_y:
        tot_reward += idle_penalty
    else:
        tot_reward += step_penalty

    tot_reward += (previous["filled"] != agent.filled) * filled_emptied

    distances = []
    prev_dist = []
    if agent.filled:
        for hole in env.holes.keys():
            distances.append(
                np.abs(hole[1] - agent.pos_x) + np.abs(hole[0] - agent.pos_y)
            )
            prev_dist.append(
                np.abs(hole[1] - previous["pos_x"])
                + np.abs(hole[0] - previous["pos_y"])
            )

    else:
        for excav in env.excavators:
            distances.append(
                np.abs(excav[1] - agent.pos_x) + np.abs(excav[0] - agent.pos_y)
            )
            prev_dist.append(
                np.abs(excav[1] - previous["pos_x"])
                + np.abs(excav[0] - previous["pos_y"])
            )

    if np.min(distances) < np.min(prev_dist):
        tot_reward += right_direction
    elif np.min(distances) > np.min(prev_dist):
        tot_reward += -right_direction

    return tot_reward
