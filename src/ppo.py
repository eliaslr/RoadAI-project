import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np


# TODO add these to hydra
BATCH_SIZE = 100
NUM_UPDATES = 10
MAX_STEPS = 100000


# TODO add a Cnn layer
# A simple NN is enough for PPO
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_l=1):
        super(SimpleNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.l1 = nn.Linear(32 * input_dim * input_dim, 128)
        self.l2 = nn.Linear(128, output_dim)

    def forward(self, X):
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
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


class PPO:
    def __init__(self, env, lr, cliprange, gamma=0.95, BATCHSIZE=10):
        self.gamma = gamma  # Discount coeff for rewards
        self.epsilon = cliprange  # Clip for actor
        self.upsilon = cliprange  # Clip for critic
        self.in_dim = env.view_dist * 2 + 1  # Square view
        # Action space is cardinal movement
        self.out_dim = 5
        # TODO test RNN
        # self.actor = nn.RNN(self.in_dim, self.out_dim)
        self.actor = SimpleNN(self.in_dim, self.out_dim)
        self.critic = SimpleNN(self.in_dim, 1)
        # self.critic = nn.RNN(self.in_dim, 1)
        self.a_optim = Adam(self.actor.parameters(), lr=lr)
        self.c_optim = Adam(self.critic.parameters(), lr=lr)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.env = env

    # Calculates discouted rewards
    def _rtgs(self, D):
        returns = np.zeros((len(D), self.env._num_agents))
        for i in reversed(range(len(D))):
            rews = D[i][3]
            if i == len(D) - 1:
                returns[i:] = rews
            else:
                for j in range(len(rews)):
                    returns[i, j] = rews[j] + returns[i + 1, j] * self.gamma
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
        return square.reduce_mean()

    # Main training loop
    # For now it only trains one ep
    # TODO add a function where we only rollout and dont update
    # TODO add model saving/loading
    def train(self):
        # Initilial Step
        self.env.step(np.zeros(len(self.env.agents)))
        curr_step = 1
        while curr_step < MAX_STEPS:
            batch = self._rollout()
            curr_step += len(batch)
            V = batch[-1][4]
            rtgs = self._rtgs(batch)
            # Calculate advantage
            A_k = rtgs - V
            # Normalize Advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # Update actor critic based on L
            for i in range(NUM_UPDATES):
                self.a_optim.zero_grad()
                self.c_optim.zero_grad()
                rand = np.random.randint(len(batch))
                (obs, actions, pi_old, rews) = batch.pop(rand)
                print(pi_old)
                _, pi = self.action(obs)
                V = self._eval(obs)
                act_loss = self._act_loss(pi, pi_old, A_k)
                cri_loss = self._cri_loss(V, rtgs)

                act_loss.requires_grad = True
                act_loss.backward()
                self.a_optim.step()

                cri_loss.requires_grad = True
                cri_loss.backward()
                self.c_optim.step()

    def _rollout(self):
        batch = []
        for i in range(BATCH_SIZE):
            obs = self.env.observation_spaces
            actions, probs = self.action(obs)
            rews = self.env.step(actions)
            batch.append((obs, actions, probs, rews))
        return batch

    def _eval(self, obs):
        V = np.zeros(len(obs))
        for i in range(len(obs)):
            V[i] = self.critic(obs[i])
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
