from collections import namedtuple, deque
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from .experiment_model import StateSpaceModel

MAX_H = 8.5
import os

_dir = os.path.dirname(os.path.abspath(__file__))
with open(
    os.path.join(_dir, "..", "parameters", "state_space_param.pickle"), "rb"
) as f:
    statespaceparam = pickle.load(f)
A, B, C, D = (
    statespaceparam["A"],
    statespaceparam["B"],
    statespaceparam["C"],
    statespaceparam["D"],
)

transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminal_flag")
)


class environment:
    def __init__(self):
        self.model = StateSpaceModel(pred_x0=False)

    def reset(self):
        self.sp = random.random()
        self.xk = np.random.rand(2)
        return [self.xk[1].item(), self.sp]

    def reward_calculation(self):
        terminated = False

        error = abs(self.xk[1] - self.sp).item()
        reward = -error * MAX_H

        if any(self.xk >= 1.0):
            terminated = True
            reward = -100

        return reward, terminated

    def step(self, action, dt=3):
        xk1, current_observation = self.model.step_pred(
            A, B, C, D, self.xk * MAX_H, action, dt, return_discrete=False
        )
        next_state = (xk1[1].item() / MAX_H, self.sp)
        self.xk = xk1 / MAX_H
        reward, terminated = self.reward_calculation()
        return next_state, reward, terminated


class QNetwork(nn.Module):
    """
    This network is estimating the Q(a,s) value, thus the input must have both state and action information.
    """

    def __init__(self, beta, input_dim, hidden_dim, layer_num):
        super(QNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        q = self.model(torch.cat([state, action], dim=1))
        return q


class VNetwork(nn.Module):
    def __init__(self, beta, input_dim, hidden_dim, layer_num):
        super(VNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        return self.model(state)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, hidden_dim, layer_num, max_action):
        super(ActorNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.mu_net = nn.Linear(hidden_dim, 1)
        self.log_std_net = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.max_action = max_action

    def forward(self, state):
        hidden_state = self.backbone(state)

        mu = self.mu_net(hidden_state)

        log_std = self.log_std_net(hidden_state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        sigma = log_std.exp()

        return mu, sigma

    def distributional_action(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)

        if reparameterize:
            # this is a random sampling method supported by the normal distribution to add additional noise for exploration
            raw_action = dist.rsample()
        else:
            raw_action = dist.sample()

        tanh_action = torch.tanh(raw_action)
        final_action = (tanh_action + 1) * self.max_action / 2

        log_prob = dist.log_prob(raw_action)
        scale = self.max_action / 2
        log_prob -= torch.log(scale * (1 - tanh_action.pow(2)) + 1e-8)

        return final_action, log_prob

    def deterministic_action(self, state):
        with torch.no_grad():
            mu, _ = self.forward(state)
            tanh_action = torch.tanh(mu)

        final_action = (tanh_action + 1) * self.max_action / 2

        return final_action


class ReplayBuffer:
    def __init__(self, capacity, state_dim=2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=bool)

    def push(self, state, action, reward, next_state, terminal):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminals[self.ptr] = terminal
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.next_states[idx], dtype=torch.float32),
            torch.tensor(self.terminals[idx]),
        )

    def __len__(self):
        return self.size


class SoftActorCritic:
    def __init__(
        self,
        v_net_arg,
        q_net_arg,
        actor_net_arg,
        replay_capa,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        reward_scale=2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_capa)
        self.batch_size = batch_size

        self.actor = ActorNetwork(**actor_net_arg)
        self.critic_1 = QNetwork(**q_net_arg)
        self.critic_2 = QNetwork(**q_net_arg)
        self.value = VNetwork(**v_net_arg)
        self.target_value = VNetwork(**v_net_arg)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, distributional=True):
        state = torch.tensor([observation], dtype=torch.float32)
        if distributional:
            action, _ = self.actor.distributional_action(state)
        else:
            action = self.actor.deterministic_action(state)
        return action.detach().item()

    def store_transition(self, state, action, reward, next_state, terminal_flag):
        self.replay_buffer.push(state, action, reward, next_state, terminal_flag)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def update(self):
        if len(self.replay_buffer) < 2 * self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        value = self.value(state_batch).view(-1)
        next_state_value = self.target_value(next_state_batch).view(-1)
        next_state_value[terminal_batch] = 0.0

        action, log_prob = self.actor.distributional_action(
            state_batch, reparameterize=False
        )
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(state_batch, action)
        q2_new_policy = self.critic_2.forward(state_batch, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_prob
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        action, log_prob = self.actor.distributional_action(
            state_batch, reparameterize=True
        )
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(state_batch, action)
        q2_new_policy = self.critic_2.forward(state_batch, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = (log_prob - critic_value).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward_batch + self.gamma * next_state_value
        q1_old_policy = self.critic_1.forward(state_batch, action_batch).view(-1)
        q2_old_policy = self.critic_2.forward(state_batch, action_batch).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def mode_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.eval()
        self.target_value.eval()

    def mode_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.value.train()
        self.target_value.train()
