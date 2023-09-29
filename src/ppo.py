import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np


# A simple FFNN is enough for PPO
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_l=1):
        super(SimpleNN, self).__init__()
        self.hl = []
        self.l1 = nn.Linear(input_dim, 64)
        for _ in range(num_hidden_l):
            self.hl.append(nn.Linear(64, 64))
        self.l3 = nn.Linear(64, output_dim)

    def forward(self, X):
        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float).flatten()
        z = nn.functional.relu(self.l1(X))
        for l in self.hl:
            z = nn.functional.relu(l(z))
        return self.l3(z)


class PPO:
    def __init__(self, env, lr, cliprange, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.clip = cliprange
        self.in_dim = env.view_dist**2 * 3
        # Action space is cardinal movement
        self.out_dim = 5
        self.nn_model = SimpleNN(self.in_dim, self.out_dim)
        self.critic = SimpleNN(self.in_dim, 1)
        self.optim = Adam(self.nn_model.parameters(), lr=lr)
        self.c_optim = Adam(self.critic.parameters(), lr=lr)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.num_updates = 3

    # Calculates discouted rewards
    def _rtgs(self, rews):
        rews = list(map(lambda x: x * self.gamma, rews))
        return torch.tensor(rews, dtype=float)

    def _eval(self, obs):
        V = np.zeros(len(obs))
        for i in range(len(obs)):
            V[i] = self.critic(obs[i])
        return torch.tensor(V)


    # Training Time Step
    # TODO Maybe add batch training
    def learn(self):
        # Rollout This prob needs a rewrite
        obs, acts, rews = self.env.step()
        # Calc V
        rtgs = self._rtgs(rews)
        V = self._eval(obs)
        A_k = rtgs - V
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)
        for i in range(len(obs)):
            V = self._eval(obs)
            curr_dist = MultivariateNormal(self.nn_model(obs[i]), self.cov_mat)
            curr_probs = curr_dist.log_prob(curr_dist.sample())
            # pi_theta / pi_theta_old
            prob_ratio = torch.exp(curr_probs - self.probs)
            # Surrogate losses
            act_loss = (-torch.min(prob_ratio * A_k,
                        (torch.clamp(prob_ratio * A_k, 1 - self.clip, 1 + self.clip) * A_k))).mean()
            critic_loss = nn.MSELoss()(V, rtgs)
            critic_loss.requires_grad = True

            self.optim.zero_grad()
            act_loss.backward(retain_graph=True)
            self.optim.step()

            self.c_optim.zero_grad()
            critic_loss.backward()
            self.c_optim.step()
            # TODO Plot loss

    # Prediction Time step
    def action(self, obs, agent):
        mu = self.nn_model(obs)
        # Create a distrubution
        dist = MultivariateNormal(mu, self.cov_mat)
        action = dist.sample()
        self.probs = dist.log_prob(action).detach()
        return np.argmax(action.detach().numpy())
