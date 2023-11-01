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
BATCH_SIZE = 750
NUM_UPDATES = 4
MAX_EPS = 600
MAX_STEPS = 5_000_000
SAVE_RATE = 25  # Save model every 25 episodes
ACTION_REPEAT = 1


# A simple NN is enough for PPO
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(ActorNetwork, self).__init__()
        # self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, output_dim)
        self.device = device

    def forward(self, X):
        # Convert to tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float, device=self.device)
        #    X =X.unsqueeze(0)
        #    X = X.unsqueeze(0)
        # z = F.relu(self.c1(X))
        # z = F.relu(self.c2(z))
        # z = z.view(z.size(0), -1)
        z = F.relu(self.l1(X))
        z = F.relu(self.l2(z))
        z = self.l3(z)
        s = nn.Softmax(dim=0)(z)
        return s


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(CriticNetwork, self).__init__()
        # self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, output_dim)
        self.device = device

    def forward(self, X):
        # Convert to tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float, device=self.device)
        #    X = X.unsqueeze(0)
        #    X = X.unsqueeze(0)
        # z = F.relu(self.c1(X))
        # z = F.relu(self.c2(z))
        # z = z.view(z.size(0), -1)
        z = F.relu(self.l1(X))
        z = F.relu(self.l2(z))
        z = self.l3(z)
        return z


class PPO:
    def __init__(
        self,
        env,
        cliprange,
        model_path,
        lr_a=0.001,
        lr_c=0.001,
        load_model=True,
        gamma=0.95,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma  # Discount coeff for rewards
        self.epsilon = cliprange  # Clip for actor
        # Saw one paper using clip Loss for critic aswell
        # self.upsilon = cliprange  # Clip for critic
        self.in_dim = 9  # Square view
        # Action space is cardinal movement
        self.out_dim = 5
        self.actor = ActorNetwork(self.in_dim, self.out_dim, device=device).to(device)
        self.critic = CriticNetwork(self.in_dim, 1, device=device).to(device)
        if load_model and os.path.isfile(model_path + "actor"):
            self.actor.load_state_dict(torch.load(model_path + "actor"))
            self.critic.load_state_dict(torch.load(model_path + "critic"))
        self.a_optim = Adam(self.actor.parameters(), lr=lr_a)
        self.c_optim = Adam(self.critic.parameters(), lr=lr_c)
        self.cov_var = torch.full(size=(self.out_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)
        self.env = env
        self.model_path = model_path

    # Calculates discouted rewards
    def _rtgs(self, rews):
        returns = np.zeros(len(rews))
        for i in reversed(range(len(rews))):
            if i == len(rews) - 1:
                returns[i] = rews[i]
            else:
                returns[i] = rews[i] + returns[i + 1] * self.gamma
        return torch.tensor(returns, dtype=float)

    # See arxiv 2103.01955 for implementation details
    def _act_loss(self, pi, pi_old, A_k):
        surr = torch.exp(pi - pi_old)
        surr2 = torch.clamp(surr, 1 - self.epsilon, 1 + self.epsilon)
        # TODO implement entropy loss
        loss = (
            -torch.min(surr * A_k, surr2 * A_k)
        ).mean()  # + self.entropy_coeff * self._entropy_loss()
        return loss

    # See arxiv 2103.01955 for implementation details
    def _cri_loss(self, V, rtgs):
        # clip = (torch.clamp(V, V_old - self.upsilon, V_old + self.upsilon) - rtgs) ** 2
        return nn.MSELoss()(V, rtgs)

    # Main training loop
    def train(self):
        torch.set_default_device(self.device)
        torch.cuda.empty_cache()
        # Initilial Step
        curr_step = 1
        curr_ep = 1
        best_mean = -np.inf
        a_loss = []
        c_loss = []
        avg_rewards = []
        while curr_step < MAX_STEPS and curr_ep < MAX_EPS:
            start = time.time()
            self.env.reset()
            self.env.step(np.zeros(len(self.env.agents)))
            b_obs, b_acts, b_probs, b_rews = self._rollout()
            avg_rewards.append(np.mean(b_rews))
            end = time.time()
            print(f"Finished episode {curr_ep} in {end - start} seconds")
            start = time.time()
            curr_ep += 1

            rtgs = self._rtgs(b_rews)
            curr_step += len(b_obs)
            V, _ = self._eval(b_obs, b_acts)
            # Calculate advantage
            A_k = rtgs - V.detach()
            # Normalize Advantage
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # Update actor critic based on L
            for _ in range(NUM_UPDATES):
                V, pi = self._eval(b_obs, b_acts)
                act_loss = self._act_loss(pi, b_probs, A_k)
                cri_loss = self._cri_loss(V, rtgs)

                a_loss.append(act_loss.item())
                c_loss.append(cri_loss.item())

                self.a_optim.zero_grad()
                act_loss.backward(retain_graph=True)
                self.a_optim.step()

                self.c_optim.zero_grad()
                cri_loss.backward()
                self.c_optim.step()
            end = time.time()
            print(f"Finished Updating the networks in {end-start}, {rtgs.mean()}")
            if avg_rewards[-1] > best_mean:
                best_mean = avg_rewards[-1]
                print(f"Saving new best model in {self.model_path}")
                if not os.path.isdir(self.model_path):
                    os.mkdir(self.model_path)
                torch.save(self.actor.state_dict(), self.model_path + "actor")
                torch.save(self.critic.state_dict(), self.model_path + "critic")

            if curr_ep % SAVE_RATE == 0:
                # self._plot_rewards(avg_rewards, a_loss, c_loss)
                torch.save(self.actor.state_dict(), self.model_path + "actor")
                torch.save(self.critic.state_dict(), self.model_path + "critic")
                self._plot_rewards(avg_rewards, a_loss, c_loss)
        return rtgs.mean()

    def _plot_rewards(self, avg_rewards, a_loss, c_loss):
        _, axs = plt.subplots(1, 3, layout="constrained")
        axs[0].set(title="ActorLoss")
        axs[1].set(title="CriticLoss")
        axs[2].set(title="MeanReward")
        axs[0].plot(np.arange(len(a_loss)), a_loss)
        axs[1].plot(np.arange(len(c_loss)), c_loss)
        axs[2].plot(np.arange(len(avg_rewards)), avg_rewards)

        plt.savefig(f"graphs/{len(avg_rewards)}.png")

    def _rollout(self):
        batch_obs = []
        batch_actions = []
        batch_probs = torch.zeros((BATCH_SIZE, self.env._num_agents))
        batch_rews = []
        for i in range(BATCH_SIZE):
            batch_obs += self.env.observation_spaces
            actions, probs = self.action(self.env.observation_spaces)
            # for _ in range(ACTION_REPEAT):
            # Fuck this is ugly
            action = list(map(lambda x: torch.argmax(x).item(), actions))
            rews = self.env.step(action)
            for j in range(self.env._num_agents):
                batch_actions.append(actions[j])
            batch_probs[i] = probs
            batch_rews += rews
        batch_probs = torch.flatten(batch_probs)
        return (batch_obs, batch_actions, batch_probs, batch_rews)

    def _eval(self, b_obs, b_acts):
        V = torch.zeros(len(b_obs), dtype=float)
        pi = torch.zeros(len(b_obs))
        for i, obs in enumerate(b_obs):
            V[i] = self.critic(obs)
            mu = self.actor(obs)
            dist = MultivariateNormal(mu, self.cov_mat)
            pi[i] = dist.log_prob(b_acts[i])
        return V, pi

    def action(self, obs):
        probs = torch.zeros(len(obs))
        actions = []
        for i in range(len(obs)):
            mu = self.actor(obs[i])
            # Create a distrubution
            dist = MultivariateNormal(mu, self.cov_mat)
            action = dist.sample()
            probs[i] = dist.log_prob(action).detach()
            actions.append(action.detach())
        return actions, probs
