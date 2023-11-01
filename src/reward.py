import numpy as np


def reward(agent, env):
    tot_reward = 0
    previous = agent.prev_agent

    # Rewards
    idle_penalty = -0.3
    step_penalty = -0.1
    collision_pen = -100
    incline_pen = 0.1
    right_direction = 5
    wrong_direction = 0
    out_of_bounds = -100

    filled = 20
    emptied = 30

    if agent.collided:
        tot_reward += collision_pen

    if agent.out_of_bounds:
        tot_reward += out_of_bounds
    elif previous["pos_x"] == agent.pos_x and previous["pos_y"] == agent.pos_y:
        tot_reward += idle_penalty
    else:
        tot_reward += step_penalty
        if env.map[previous["pos_y"], previous["pos_x"]] < agent._ground:
            tot_reward += (
                env.map[previous["pos_y"], previous["pos_x"]] - agent._ground
            ) * incline_pen

    if agent.filled and not previous["filled"]:
        tot_reward += filled
    if not agent.filled and previous["filled"]:
        tot_reward += emptied

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
        tot_reward += wrong_direction
    return tot_reward
