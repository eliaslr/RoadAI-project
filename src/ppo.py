import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# TODO add these to hydra
BATCH_SIZE = 1000
NUM_UPDATES = 3
MAX_EPS = 200
MAX_STEPS = 1_000_000
SAVE_RATE = 5  # Save model every 5 episodes


# TODO add a Cnn layer
# A simple NN is enough for PPO
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.l1 = nn.Linear(32 * input_dim * input_dim, 128)
        self.l2 = nn.Linear(128, output_dim)

    def forward(self, X):
        # Convert to tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float)
            X = X.unsqueeze(0)
            X = X.unsqueeze(0)
        z = F.relu(self.c1(X))
        # z = F.max_pool2d(z, 2)
        z = F.relu(self.c2(z))
        # z = F.max_pool2d(z, 2)
        z = z.view(z.size(0), -1)
        z = F.relu(self.l1(z))
        z = self.l2(z)
        s = nn.Softmax(dim=1)(z)
        return s


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CriticNetwork, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.l1 = nn.Linear(32 * input_dim * input_dim, 128)
        self.l2 = nn.Linear(128, output_dim)

    def forward(self, X):
        # Convert to tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float)
            X = X.unsqueeze(0)
            X = X.unsqueeze(0)
        z = F.relu(self.c1(X))
        # z = F.max_pool2d(z, 2)
        z = F.relu(self.c2(z))
        # z = F.max_pool2d(z, 2)
        z = z.view(z.size(0), -1)
        z = F.relu(self.l1(z))
        z = self.l2(z)
        return z


class PPO:
    def __init__(self, env, lr, cliprange, model_path, load_model=False, gamma=0.95):
        self.gamma = gamma  # Discount coeff for rewards
        self.epsilon = cliprange  # Clip for actor
        # Saw one paper using clip Loss for critic aswell
        # self.upsilon = cliprange  # Clip for critic
        self.in_dim = env.view_dist * 2 + 1  # Square view
        # Action space is cardinal movement
        self.out_dim = 5
        self.actor = ActorNetwork(self.in_dim, self.out_dim)
        self.critic = CriticNetwork(self.in_dim, 1)
        if load_model and os.path.isfile(model_path + "actor"):
            self.actor.load_state_dict(torch.load(model_path + "actor"))
            self.critic.load_state_dict(torch.load(model_path + "critic"))
        self.a_optim = Adam(self.actor.parameters(), lr=lr)
        self.c_optim = Adam(self.critic.parameters(), lr=lr)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.env = env
        self.model_path = model_path

    # Calculates discouted rewards
    def _rtgs(self, rews):
        returns = np.zeros((len(rews), self.env._num_agents))
        for i in reversed(range(len(rews))):
            if i == len(rews) - 1:
                returns[i:] = rews[i]
            else:
                for j in range(len(rews[i])):
                    returns[i, j] = rews[i][j] + returns[i + 1, j] * self.gamma
        return torch.tensor(returns, dtype=float)

    # See arxiv 2103.01955 for implementation details
    def _act_loss(self, pi, pi_old, A_k):
        surr = torch.tensor(np.exp(pi - pi_old))
        surr2 = torch.clamp(surr, 1 - self.epsilon, 1 + self.epsilon)
        # TODO implement entropy loss
        loss = (
            -torch.min(surr * A_k, surr2 * A_k)
        ).mean()  # + self.entropy_coeff * self._entropy_loss()
        return loss

    # See arxiv 2103.01955 for implementation details
    def _cri_loss(self, V, rtgs):
        square = (V - rtgs) ** 2
        # clip = (torch.clamp(V, V_old - self.upsilon, V_old + self.upsilon) - rtgs) ** 2
        return square.mean()

    # Main training loop
    # TODO add a function where we only rollout and dont update
    def train(self):
        # Initilial Step
        curr_step = 1
        curr_ep = 1
        best_mean = 0
        loss = []
        while curr_step < MAX_STEPS and curr_ep < MAX_EPS:
            start = time.time()
            self.env.reset()
            self.env.step(np.zeros(len(self.env.agents)))
            b_obs, b_acts, b_probs, b_rews = self._rollout()
            end = time.time()
            print(f"Finished episode {curr_ep} in {end - start} seconds")
            start = time.time()
            curr_ep += 1
            curr_step += len(b_obs)
            V = self._eval(b_obs)
            rtgs = self._rtgs(b_rews)
            # Calculate advantage
            A_k = rtgs - V
            # Normalize Advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # Update actor critic based on L
            for i in range(NUM_UPDATES):
                pi = np.zeros((len(b_obs), self.env._num_agents))
                for i in range(BATCH_SIZE):
                    _, pi[i:] = self.action(b_obs[i])
                V = self._eval(b_obs)
                print(V)
                act_loss = self._act_loss(pi, b_probs, A_k)
                cri_loss = self._cri_loss(V, rtgs)
                print(act_loss.item(), cri_loss.item())

                self.a_optim.zero_grad()
                act_loss.requires_grad = True
                act_loss.backward()
                self.a_optim.step()

                self.c_optim.zero_grad()
                cri_loss.requires_grad = True
                cri_loss.backward()
                self.c_optim.step()
            end = time.time()
            print(
                f"Finished Updating the networks in {end-start}, {self.env.avg_rewards[-1]}"
            )
            if self.env.avg_rewards[-1] > best_mean:
                best_mean = self.env.avg_rewards[-1]
                print(f"Saving new best model in {self.model_path}")
                torch.save(self.actor.state_dict(), self.model_path + "actor")
                torch.save(self.critic.state_dict(), self.model_path + "critic")
        self._plot_rewards(curr_ep, loss)

    def _plot_rewards(self, curr_ep, loss):
        plt.plot(np.arange(len(self.env.avg_rewards)), self.env.avg_rewards)
        plt.savefig(f"{curr_ep}_rew.png")
        plt.plot(np.arange((curr_ep - 1) * NUM_UPDATES), loss)
        plt.savefig(f"{curr_ep}_loss.png")

    def _rollout(self):
        batch_obs = []
        batch_actions = np.zeros((BATCH_SIZE, self.env._num_agents))
        batch_probs = np.zeros((BATCH_SIZE, self.env._num_agents))
        batch_rews = []
        for i in range(BATCH_SIZE):
            batch_obs.append(self.env.observation_spaces)
            actions, probs = self.action(self.env.observation_spaces)
            rews = self.env.step(actions)
            batch_actions[i:] = actions
            batch_probs[i:] = probs
            batch_rews.append(rews)
        return (batch_obs, batch_actions, batch_probs, batch_rews)

    def _eval(self, b_obs):
        V = np.zeros((len(b_obs), self.env._num_agents))
        for i in range(len(b_obs)):
            for j in range(self.env._num_agents):
                V[i, j] = self.critic(b_obs[i][j])
        return torch.tensor(V, dtype=float)

    def action(self, obs):
        probs = np.zeros(len(obs))
        actions = np.zeros(len(obs))
        for i in range(len(obs)):
            mu = self.actor(obs[i])
            # Create a distrubution
            dist = MultivariateNormal(mu, self.cov_mat)
            action = dist.sample()
            probs[i] = dist.log_prob(action).detach().numpy()
            actions[i] = np.argmax(action.detach().numpy())
        return actions, probs
