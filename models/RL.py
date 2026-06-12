from collections import namedtuple, deque
import math
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import Beta

from .experiment_model import StateSpaceModel


MAX_H = 8.5
import os
_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, "..", "parameters", "state_space_param.pickle"), "rb") as f:
    statespaceparam = pickle.load(f)
A, B, C, D = statespaceparam['A'], statespaceparam['B'], statespaceparam['C'], statespaceparam['D']

transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal_flag'))


class environment():
    def __init__(self):
        self.model = StateSpaceModel(pred_x0=False)

    def reset(self):
        self.sp = random.random()
        self.xk = np.random.rand(2)
        return [self.xk[1].item(), self.sp]

    def reward_calculation(self):
        terminated = False

        error = abs(self.xk[1] - self.sp).item()
        reward = -error*MAX_H

        if any(self.xk >= 1.0): 
            terminated = True
            reward = -100

        return reward, terminated

    def step(self, action, dt=3):
        xk1, current_observation = self.model.step_pred(A, B, C, D, self.xk*MAX_H, action, dt, return_discrete=False)
        next_state = (xk1[1].item()/MAX_H, self.sp)
        self.xk = xk1/MAX_H
        reward, terminated = self.reward_calculation()
        return next_state, reward, terminated


class SingleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, output_dim):
        super(SingleNetwork, self).__init__()

        layers = []
        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        #hidden layer
        if layer_num >= 2:
            for i in range(layer_num - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
        #output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)
    

class CommonBodyNetwork(nn.Module):
    def __init__(self, input_dim, com_hidden_dim, com_layer_num, low_feature_dim, head_hidden_dim, head_layer_num, action_output_dim, value_output_dim):
        super(CommonBodyNetwork, self).__init__()

        common_layers = []
        common_layers.append(nn.Linear(input_dim, com_hidden_dim))
        common_layers.append(nn.ReLU())
        if com_layer_num >= 2:
            for i in range(com_layer_num - 1):
                common_layers.append(nn.Linear(com_hidden_dim, com_hidden_dim))
                common_layers.append(nn.ReLU())
        common_layers.append(nn.Linear(com_hidden_dim, low_feature_dim))
        common_layers.append(nn.ReLU())

        critic_layers = []
        critic_layers.append(nn.Linear(low_feature_dim, head_hidden_dim))
        critic_layers.append(nn.ReLU())
        if head_layer_num >= 2:
            for i in range(head_layer_num - 1):
                critic_layers.append(nn.Linear(head_hidden_dim, head_hidden_dim))
                critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(head_hidden_dim, value_output_dim))

        actor_layers = []
        actor_layers.append(nn.Linear(low_feature_dim, head_hidden_dim))
        actor_layers.append(nn.ReLU())
        if head_layer_num >= 2:
            for i in range(head_layer_num - 1):
                actor_layers.append(nn.Linear(head_hidden_dim, head_hidden_dim))
                actor_layers.append(nn.ReLU())
        actor_layers.append(nn.Linear(head_hidden_dim, action_output_dim))

        self.common = nn.Sequential(*common_layers)
        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, x):
        features    = self.common(x)
        logits      = self.actor(features)       # raw linear output, unbounded
        state_value = self.critic(features)
        
        alpha = F.softplus(logits[:, 0:1]) + 1   # softplus here, not in the layer
        beta  = F.softplus(logits[:, 1:2]) + 1
        
        return (alpha, beta), state_value
    

class DQN():
    def __init__(self, network_arg, lr, replay_capa):
        self.Q_func = SingleNetwork(**network_arg)
        self.target_func = SingleNetwork(**network_arg)
        self.target_func.load_state_dict(self.Q_func.state_dict())

        self.replay_buffer = deque([], maxlen=replay_capa)

        self.optimizer = optim.AdamW(self.Q_func.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.loss_history = []

    def epsilon_greedy_action(self, step, state):
        eps = 0.01 + (0.9 - 0.01) * math.exp(-1. * step / 500000)
        prob = random.random()

        state = torch.Tensor(state)
        if prob >= eps:
            with torch.no_grad():
                action = self.Q_func(state).max(0).indices.item()
        else:
            output_dim = self.Q_func.model[-1].out_features
            action = random.randint(0, output_dim-1)
        
        return action
    
    def greedy_action(self, state):
        state = torch.Tensor(state)
        with torch.no_grad():
            action = self.Q_func(state).max(0).indices.item()
        return action
    
    def store_transition(self, state, action, reward, next_state, terminal_flag):
        history = transition(state, action, reward, None if terminal_flag else next_state, terminal_flag)
        self.replay_buffer.append(history)

    def update(self, minibatch_size):
        minibatch = random.sample(self.replay_buffer, minibatch_size)
        minibatch = transition(*zip(*minibatch))

        state_batch = torch.tensor(minibatch.state, dtype=torch.float32)
        action_batch = torch.tensor(minibatch.action, dtype=torch.long).unsqueeze(-1)
        reward_batch = torch.tensor(minibatch.reward, dtype=torch.float32)
        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, minibatch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.Tensor([s for s in minibatch.next_state if s is not None])

        q = self.Q_func(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(minibatch_size)
        with torch.no_grad():
            next_state_values[non_terminal_mask] = self.target_func(non_final_next_states).max(1)[0]

        y = reward_batch + 0.99*next_state_values

        loss = self.criterion(q.squeeze(), y)
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_func.parameters(), max_norm=5)
        self.optimizer.step()

        target_net_state_dict = self.target_func.state_dict()
        q_func_state_dict = self.Q_func.state_dict()
        for key in q_func_state_dict:
            target_net_state_dict[key] = q_func_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
        self.target_func.load_state_dict(target_net_state_dict)


class PPO():
    def __init__(self, network_arg, lr, max_speed):
        self.network = CommonBodyNetwork(**network_arg)
        
        separate_lr = [
            {"params": self.network.common.parameters(), "lr": lr["common"]},
            {"params": self.network.critic.parameters(), "lr": lr["critic"]},
            {"params": self.network.actor.parameters(), "lr": lr["actor"]}
        ]
        self.optimizer = optim.Adam(separate_lr)

        self.max_speed = max_speed

        self.transition_storage = []
        self.old_pi = []

        self.tot_loss_history, self.actor_loss_history, self.critic_loss_history, self.entropy_loss_history = [], [], [], []
        self.gradient_history = []

    def _dist(self, alpha, beta):
    # alpha, beta must be > 0; softplus ensures this
        return Beta(alpha, beta)

    def distributional_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        with torch.no_grad():
            (alpha, beta), value = self.network(state)
        dist   = self._dist(alpha.squeeze(0), beta.squeeze(0))
        action = dist.sample() * self.max_speed          # (0,1) → (0, 36)
        self.old_pi.append(dist.log_prob(action / self.max_speed))
        return action.item(), value

    def deterministic_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        with torch.no_grad():
            (alpha, beta), _ = self.network(state)
        # mode = (α-1)/(α+β-2) for α,β > 1; falls back to mean α/(α+β) otherwise
        alpha, beta = alpha.squeeze(0), beta.squeeze(0)
        mode = (alpha - 1) / (alpha + beta - 2)
        return (mode * self.max_speed).item()

    def store_transition(self, state, action, reward, next_state, terminal_flag):
        history = transition(state, action, reward, next_state, terminal_flag)
        self.transition_storage.append(history)

    def update(self, GAMMA, LAMBDA, epochs, batch_size, epsilon, beta_coef):
        batch = transition(*zip(*self.transition_storage))
        self.transition_storage.clear()

        batch_state      = torch.Tensor(batch.state)
        batch_action     = torch.Tensor(batch.action)
        batch_reward     = torch.Tensor(batch.reward)
        batch_next_state = torch.stack([torch.Tensor(s) for s in batch.next_state])
        batch_done       = torch.Tensor([float(d) for d in batch.terminal_flag])

        with torch.no_grad():
            _, batch_values      = self.network(batch_state)
            _, batch_next_values = self.network(batch_next_state)
        batch_values      = batch_values.squeeze()
        batch_next_values = batch_next_values.squeeze()

        # GAE
        adv, ret = 0, 0.0
        advantages, returns = [], []
        for i in reversed(range(len(batch_reward))):
            if batch_done[i]:
                delta = batch_reward[i] - batch_values[i]
                adv   = delta
                ret   = 0.0
            else:
                delta = batch_reward[i] + GAMMA * batch_next_values[i] - batch_values[i]
                adv   = delta + GAMMA * LAMBDA * adv
            advantages.append(adv)
            ret = batch_reward[i] + GAMMA * ret
            returns.append(ret)

        advantages.reverse(); returns.reverse()
        advantages = torch.Tensor(advantages)
        returns    = torch.Tensor(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.stack(self.old_pi[:len(batch_reward)]).detach()
        self.old_pi.clear()

        tot_l, act_l, cri_l, ent_l = [], [], [], []

        for _ in range(epochs):
            for idx in torch.randperm(len(batch_reward)).split(batch_size):
                mb_state     = batch_state[idx]
                mb_action    = batch_action[idx]
                mb_advantage = advantages[idx]
                mb_return    = returns[idx]
                mb_old_lp    = old_log_probs[idx]

                (alpha, beta), values = self.network(mb_state)
                values = values.squeeze()
                dist   = self._dist(alpha, beta)

                # normalize action back to (0,1) for Beta log_prob
                mb_action_normalized = (mb_action / self.max_speed).clamp(1e-6, 1 - 1e-6)
                new_log_prob = dist.log_prob(mb_action_normalized)

                ratio   = (new_log_prob - mb_old_lp).exp()
                clipped = ratio.clamp(1 - epsilon, 1 + epsilon)

                actor_loss   = -torch.min(ratio * mb_advantage, clipped * mb_advantage).mean()
                critic_loss  = nn.MSELoss()(mb_return.detach(), values)
                entropy_loss = -dist.entropy().mean()
                loss         = actor_loss + 0.5 * critic_loss + beta_coef * entropy_loss

                tot_l.append(loss.item()); act_l.append(actor_loss.item())
                cri_l.append(critic_loss.item()); ent_l.append(entropy_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.gradient_history.append(
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                )
                self.optimizer.step()

        self.tot_loss_history.append(np.mean(tot_l))
        self.actor_loss_history.append(np.mean(act_l))
        self.critic_loss_history.append(np.mean(cri_l))
        self.entropy_loss_history.append(np.mean(ent_l))