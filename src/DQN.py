"""
i had nothing to do so i attempted dqn

"""
import numpy as np

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from enviroment import RoadEnv
import reward
import agent

# 5 actions per agent, if this is changed in the future it needs to change here as well
actions_per_agent = 5

env = RoadEnv(reward.reward)

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# to use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', {"state","action","next_state","reward"})

class ReplayMemory():
    """
        storing things that happened for training the network later

        faster than using only the most recent state/action tuple
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)


    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = actions_per_agent

state = torch.tensor(env.reset(min_h = 70, max_h = 70, min_w = 70, max_w = 70).flatten())

states = torch.from_numpy(np.array([torch.cat((*agent.info,state)) for agent in env.agents]))

# +2 because we need to include info about the agent
n_observations = len(state) + 2

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(states):
    global steps_done
    sample = random.random()

    #epsilon for policy
    eps_threshhold = EPS_END + (EPS_START-EPS_END)*math.exp(-1*steps_done/EPS_DECAY)

    steps_done += 1

    # were using epsilon-greedy to decide action
    if sample > eps_threshhold:
        with torch.no_grad():
            actions = []
            for state in states:
                actions.append(policy_net(state).max(1)[1].view(1, 1))
    else:
        actions = []
        for i in range(len(env.agents)):
            sample = random.randint(0,4)
            actions.append(range(5)[sample])

    return torch.tensor(actions, dtype = torch.int64)

episode_durations = []

def plot_durations(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype = torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # 100 episode average plot as well to get a smoother curve
    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait = True)
        else:
            display.display(plt.gcf())



def optimize_model():
    # TODO: Make this work for multi agents, this is for single agent
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    #transpose the batch for some reason
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype = torch.bool)

    non_final_next_states = torch.cat([s for s in batch[2] if s is not None])

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])

    # state_batch = torch.tensor(state_batch, dtype = torch.float32)
    # Compute Q(s_t, a), we compute Q(s_t) then we select action. these are the actions we wouldve
    # taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values*GAMMA) + reward_batch

    # hubert loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

def main(render):
    for i_episode in range(num_episodes):
        state = torch.tensor(env.reset(min_h = 70, max_h = 70, min_w = 70, max_w = 70, curr_ep = env.curr_ep).flatten(), dtype = torch.float32, device = device).unsqueeze(0)
        states = torch.from_numpy(np.array([torch.cat((agent.info,state),1) for agent in env.agents], dtype = np.float32))

        for t in count():
            actions = select_action(states)
            observation, reward, terminated, truncated = env.step_deep(actions, render)
            reward = torch.tensor([reward], device = device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(observation).type(torch.float32).unsqueeze(0)
                next_states = torch.from_numpy(np.array([torch.cat((agent.info,next_state),1) for agent in env.agents]))
            for i in range(len(actions)):
                memory.push(states[i], actions[i].view(1,1), next_states[i], reward.view(1,1))

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t+1)
                plot_durations()
                break

    print("Complete")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


def parse_args():
  """Parse command line argument."""

  parser = argparse.ArgumentParser("Train roadAI MARL.")
  parser.add_argument("-r", "--render", help="How to render the model while training (console, pygame). Leave blank to not render")

  return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.render)
