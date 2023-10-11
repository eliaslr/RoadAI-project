import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

# TODO add these to hydra
BATCH_SIZE = 100
NUM_UPDATES = 10
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
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
        z = nn.functional.relu(self.l1(X))
        for l in self.hl:
            z = nn.functional.relu(l(z))
        return self.l3(z)


class PPO:
    def __init__(self, env, lr, cliprange, gamma=0.95, BATCHSIZE=10):
        self.gamma = gamma  # Discount coeff for rewards
        self.epsilon = cliprange  # Clip for actor
        self.upsilon = cliprange  # Clip for critic
        self.in_dim = (env.view_dist * 2 + 1) ** 2  # Square view
        # Action space is cardinal movement
        self.out_dim = 5
        # TODO test RNN
        # self.actor = nn.RNN(self.in_dim, self.out_dim)
        self.actor = SimpleNN(self.in_dim, self.out_dim)
        self.critic = SimpleNN(self.in_dim, 1)
        self.a_optim = Adam(self.actor.parameters(), lr=lr)
        self.c_optim = Adam(self.critic.parameters(), lr=lr)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.env = env

    # Calculates discouted rewards
    def _rtgs(self, D):
        returns = []
        for i in range(len(D)):
            rews = D[i][3]
            returns.append(self.gamma * rews)
        returns = np.array(returns)
        return torch.tensor(returns, dtype=float)

    # See arxiv 2103.01955 for implementation details
    def _act_loss(self, pi, pi_old, A_k):
        surr = torch.tensor(np.exp(pi - pi_old))
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

    # Main training loop
    # For now it only trains one ep
    # TODO add a function where we only rollout and dont update
    def train(self, render_mode=None):
        # Initilial Step
        self.env.step(np.zeros(len(self.env.agents)))
        curr_step = 1
        while curr_step < MAX_STEPS:
            D = []
            # Rollout
            for i in range(BATCH_SIZE):
                actions = np.zeros(len(self.env.agents))
                probs = np.zeros(len(self.env.agents))
                rews = np.zeros(len(self.env.agents))
                obs = self.env.observation_spaces
                actions, probs = self.action(obs)
                V = self._eval(obs)
                rews = self.env.step(actions)
                if render_mode is not None:
                    self.env.render(render_mode)
                D.append((obs, actions, probs, rews, V))
                curr_step += 1
            V = D[-1][4]
            rtgs = self._rtgs(D)
            # Calculate advantage
            A_k = rtgs - V
            # Normalize Advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for i in range(NUM_UPDATES):
                rand = np.random.randint(len(D))
                (obs, actions, pi_old, rews, v_old) = D.pop(rand)
                _, pi = self.action(obs)
                V = self._eval(obs)
                act_loss = self._act_loss(pi, pi_old, A_k)
                cri_loss = self._cri_loss(V, v_old, rtgs)
                print(rtgs)
                print(f"ALOSS:{act_loss.item()} CLOSS:{ cri_loss.item()}")

                self.a_optim.zero_grad()
                act_loss.requires_grad = True
                act_loss.backward()
                self.a_optim.step()

                self.c_optim.zero_grad()
                cri_loss.requires_grad = True
                cri_loss.backward()
                self.c_optim.step()

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
            actions[i] = np.argmax(action.numpy())
            probs[i] = dist.log_prob(action).detach().numpy()
        return actions, probs
