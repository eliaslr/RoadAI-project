import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

# TODO add these to hydra
BATCH_SIZE = 10
NUM_UPDATES = 3
MAX_STEPS = 1000


# TODO add a Cnn layer
# A simple NN is enough for PPO
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
    def __init__(self, env, lr, cliprange, gamma=0.95, BATCHSIZE=10):
        self.gamma = gamma
        self.clip = cliprange
        # self.in_dim = env.view_dist**2 * 3
        self.in_dim = 16
        # Action space is cardinal movement
        self.out_dim = 5
        self.actor = nn.RNN(self.in_dim, self.out_dim)
        self.critic = SimpleNN(self.in_dim, 1)
        self.a_optim = Adam(self.actor.parameters(), lr=lr)
        self.c_optim = Adam(self.critic.parameters(), lr=lr)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.num_updates = 3

    # Calculates discouted rewards
    def _rtgs(self, rews):
        returns = np.zeros(len(rews))
        for i in range(len(rews)):
            returns[i] = self.gamma * np.sum(rews[:i])
        return torch.tensor(returns, dtype=float)

    # See arxiv 2103.01955 for implementation details
    def _act_loss(self, pi, pi_old, A_k):
        surr = torch.exp(pi - pi_old)
        surr2 = torch.clamp(surr, 1 - self.epsilon, 1 + self.epsilon)
        # TODO implement entropy loss
        loss = (
            -torch.min(surr, surr2)
        ).mean()  # + self.entropy_coeff * self._entropy_loss()
        return loss

    # See arxiv 2103.01955 for implementation details
    def _cri_loss(self, V, V_old, rtgs):
        square = (V - rtgs) ** 2
        clip = (torch.clamp(V, V_old - self.upsilon, V_old + self.upsilon) - rtgs) ** 2
        return torch.max(square, clip).mean()

    # Training Time Step
    # TODO Maybe add batch training
    def train(self, render_mode=None):
        curr_step = 0
        while curr_step < MAX_STEPS:
            D = []
            # Rollout
            for i in range(BATCH_SIZE):
                actions = np.zeros(len(self.env.agents))
                probs = np.zeros(len(self.env.agents))
                rews = np.zeros(len(self.env.agents))
                obs = self.env.observation_spaces
                V = np.zeros(len(self.env.agents))
                actions, probs = self.action(obs)
                V = self._eval(obs)
                rews = self.env.step(actions)
                if render_mode is not None:
                    self.env.render(render_mode)
                rtgs = self._rtgs(rews)
                A_k = rtgs - V
                # Normalize advantage
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
                D.append((obs, actions, probs, V, A_k, rtgs))
                curr_step += 1
            for i in range(NUM_UPDATES):
                rand = np.random.randint(len(D))
                (obs, actions, pi_old, v_old, A_k, rtgs) = D.remove(rand)
                _, pi = self.action(obs)
                V = self._eval(obs)
                act_loss = self._act_loss(pi, pi_old, A_k)
                cri_loss = self._cri_loss()

                self.a_optim.zero_grad()
                act_loss.backward()
                self.a_optim.step()

                self.c_optim.zero_grad()
                cri_loss.backward()
                self.c_optim.step()

    def _eval(self, obs):
        V = np.zeros(len(obs))
        for i in range(len(obs)):
            V[i] = self.critic(obs[i])
        return V

    def action(self, obs):
        probs = np.zeros(len(obs))
        actions = np.zeros(len(obs))
        for i in range(len(obs)):
            mu = self.actor(torch.tensor(obs[i]))
            # Create a distrubution
            dist = MultivariateNormal(mu, self.cov_mat)
            actions[i] = dist.sample()
            probs[i] = dist.log_prob(action).detach()
        return actions, probs
