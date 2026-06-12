from collections import namedtuple, deque
import math
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

# One transition within an episode step
transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminal_flag")
)


# ---------------------------------------------------------------------------
# Environment  (unchanged from original)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class LSTMBodyNetwork(nn.Module):
    """
    Architecture
    ─────────────────────────────────────────────
    Input  → MLP encoder  →  LSTM  →  actor head  (α, β)
                                   →  critic head (V)

    The LSTM replaces the common MLP body so that the policy can integrate
    information over time and overcome partial observability.
    """

    def __init__(
        self,
        input_dim,  # obs dimension
        enc_hidden_dim,  # MLP encoder width
        enc_layer_num,  # MLP encoder depth (≥1)
        lstm_hidden_dim,  # LSTM hidden size
        lstm_num_layers,  # stacked LSTM layers
        head_hidden_dim,  # actor/critic MLP head width
        head_layer_num,  # actor/critic MLP head depth (≥1)
        action_output_dim,  # 2  (α raw, β raw)
        value_output_dim,
    ):  # 1
        super().__init__()

        # ── MLP encoder ────────────────────────────────────────────────────
        enc_layers = [nn.Linear(input_dim, enc_hidden_dim), nn.ReLU()]
        for _ in range(enc_layer_num - 1):
            enc_layers += [nn.Linear(enc_hidden_dim, enc_hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)
        enc_out_dim = enc_hidden_dim

        # ── LSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=enc_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # Orthogonal init for recurrent weights — standard for LSTM stability
        for name, p in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Forget-gate bias = 1: encourages long-term memory early in training
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)

        # ── Actor head ─────────────────────────────────────────────────────
        actor_layers = [nn.Linear(lstm_hidden_dim, head_hidden_dim), nn.ReLU()]
        for _ in range(head_layer_num - 1):
            actor_layers += [nn.Linear(head_hidden_dim, head_hidden_dim), nn.ReLU()]
        actor_layers.append(nn.Linear(head_hidden_dim, action_output_dim))
        self.actor = nn.Sequential(*actor_layers)

        # ── Critic head ────────────────────────────────────────────────────
        critic_layers = [nn.Linear(lstm_hidden_dim, head_hidden_dim), nn.ReLU()]
        for _ in range(head_layer_num - 1):
            critic_layers += [nn.Linear(head_hidden_dim, head_hidden_dim), nn.ReLU()]
        critic_layers.append(nn.Linear(head_hidden_dim, value_output_dim))
        self.critic = nn.Sequential(*critic_layers)

    # ------------------------------------------------------------------
    # forward – supports both single-step (inference) and sequence (update)
    # ------------------------------------------------------------------
    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len, input_dim)  OR  (batch, input_dim) for single step
        hidden : tuple of (h, c) each (num_layers, batch, lstm_hidden_dim), or None
        returns: (alpha, beta), state_value, (h_n, c_n)
        """
        squeeze = False
        if x.dim() == 2:  # single step: add seq dim
            x = x.unsqueeze(1)
            squeeze = True

        B_sz, T, _ = x.shape

        # encode every timestep independently
        enc = self.encoder(x.reshape(B_sz * T, -1)).reshape(B_sz, T, -1)

        lstm_out, hidden_out = self.lstm(enc, hidden)  # (B, T, H)

        logits = self.actor(lstm_out)  # (B, T, 2)
        state_value = self.critic(lstm_out)  # (B, T, 1)

        # Clamp logits before softplus to prevent inf/nan from exploding LSTM output
        logits = logits.clamp(-10.0, 10.0)
        alpha = F.softplus(logits[..., 0:1]) + 1
        beta = F.softplus(logits[..., 1:2]) + 1

        if squeeze:
            alpha = alpha.squeeze(1)  # (B, 1)
            beta = beta.squeeze(1)
            state_value = state_value.squeeze(1)  # (B, 1)

        return (alpha, beta), state_value, hidden_out

    def init_hidden(self, batch_size=1, device=None):
        """Return zero hidden state."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device
        )
        c = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device
        )
        return (h, c)


# ---------------------------------------------------------------------------
# Recurrent PPO
# ---------------------------------------------------------------------------
class RecurrentPPO:
    """
    LSTM-based Recurrent PPO with Beta-distribution actor.

    Episode data is stored as complete sequences so that BPTT is done
    over the full rollout length inside update().

    Public interface is identical to the original PPO class:
        distributional_action(state) → action, value
        deterministic_action(state)  → action
        store_transition(...)
        update(...)
        reset_hidden()               ← NEW: call at the start of every episode
    """

    def __init__(self, network_arg, lr, max_speed):
        self.network = LSTMBodyNetwork(**network_arg)
        self.max_speed = max_speed

        # Initialise actor output layer to zero so α≈β≈1 (uniform) at start
        nn.init.zeros_(self.network.actor[-1].weight)
        nn.init.zeros_(self.network.actor[-1].bias)

        separate_lr = [
            {"params": self.network.encoder.parameters(), "lr": lr["common"]},
            {"params": self.network.lstm.parameters(), "lr": lr["common"]},
            {"params": self.network.critic.parameters(), "lr": lr["critic"]},
            {"params": self.network.actor.parameters(), "lr": lr["actor"]},
        ]
        self.optimizer = optim.Adam(separate_lr)

        # ── rollout buffers ────────────────────────────────────────────────
        # Each element of episode_buffer is a list of transitions for one episode.
        # current_episode collects the ongoing episode.
        self.episode_buffer = []  # list of completed episode lists
        self.current_episode = []  # list of transition namedtuples
        self.old_log_probs = []  # log π_old per step, appended in distributional_action

        # ── recurrent hidden state (maintained across steps in one episode) ─
        self.hidden = self.network.init_hidden(batch_size=1)

        # ── diagnostics ───────────────────────────────────────────────────
        self.tot_loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_loss_history = []
        self.gradient_history = []

    # ------------------------------------------------------------------
    # Hidden-state management
    # ------------------------------------------------------------------
    def reset_hidden(self):
        """Must be called at the beginning of each new episode."""
        self.hidden = self.network.init_hidden(batch_size=1)

    def _detach_hidden(self, hidden):
        return (hidden[0].detach(), hidden[1].detach())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dist(self, alpha, beta):
        return Beta(alpha, beta)

    # ------------------------------------------------------------------
    # Action selection (online, single step)
    # ------------------------------------------------------------------
    def distributional_action(self, state):
        """
        Sample an action from the Beta policy using the current hidden state.
        Appends log π_old to self.old_log_probs for later use in update().
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, input_dim)
        with torch.no_grad():
            (alpha, beta), value, self.hidden = self.network(state_t, self.hidden)
        self.hidden = self._detach_hidden(self.hidden)

        alpha = alpha.squeeze()  # (1,1) → scalar-safe
        beta = beta.squeeze()
        dist = self._dist(alpha, beta)
        action = dist.sample() * self.max_speed
        # squeeze to 0-d so torch.stack later gives (total_steps,) not (total_steps,1)
        self.old_log_probs.append(dist.log_prob(action / self.max_speed).squeeze())
        return action.item(), value

    def deterministic_action(self, state):
        """Greedy action (mode of Beta); does NOT update old_log_probs."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            (alpha, beta), _, self.hidden = self.network(state_t, self.hidden)
        self.hidden = self._detach_hidden(self.hidden)

        alpha = alpha.squeeze()
        beta = beta.squeeze()
        mode = (alpha - 1) / (alpha + beta - 2)
        return (mode * self.max_speed).item()

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, terminal_flag):
        """Append one step to the current episode buffer."""
        self.current_episode.append(
            transition(state, action, reward, next_state, terminal_flag)
        )
        if terminal_flag:
            self.episode_buffer.append(self.current_episode)
            self.current_episode = []
            self.reset_hidden()  # auto-reset so next episode starts clean

    def flush_episode(self):
        """
        Call at the end of a truncated (non-terminal) epoch to store the
        partial episode so it is included in the next update().
        """
        if self.current_episode:
            self.episode_buffer.append(self.current_episode)
            self.current_episode = []

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self, GAMMA, LAMBDA, epochs, batch_size, epsilon, beta_coef):
        """
        Recurrent PPO update.

        All stored episodes are padded into a (num_episodes, max_T, ...) batch.
        The LSTM is unrolled over the full sequence length with a zero initial
        hidden state (episode boundaries are already respected by construction).
        PPO losses are computed only on valid (non-padded) timesteps.
        """
        # ── 1. Collect all episodes (including any unfinished partial episode) ─
        self.flush_episode()
        all_episodes = self.episode_buffer
        self.episode_buffer = []

        if not all_episodes:
            return

        num_ep = len(all_episodes)
        lengths = [len(ep) for ep in all_episodes]
        max_T = max(lengths)
        input_dim = len(all_episodes[0][0].state)

        # ── 2. Build padded tensors ────────────────────────────────────────
        states = torch.zeros(num_ep, max_T, input_dim)
        next_states = torch.zeros(num_ep, max_T, input_dim)
        actions = torch.zeros(num_ep, max_T)
        rewards = torch.zeros(num_ep, max_T)
        dones = torch.zeros(num_ep, max_T)
        mask = torch.zeros(num_ep, max_T, dtype=torch.bool)  # True = valid

        for i, ep in enumerate(all_episodes):
            T_i = len(ep)
            for t, tr in enumerate(ep):
                states[i, t] = torch.FloatTensor(tr.state)
                next_states[i, t] = torch.FloatTensor(tr.next_state)
                actions[i, t] = float(tr.action)
                rewards[i, t] = float(tr.reward)
                dones[i, t] = float(tr.terminal_flag)
            mask[i, :T_i] = True

        # ── 3. Compute old log-probs (already stored per step in order) ────
        old_log_probs_flat = torch.stack(self.old_log_probs).detach()  # (total_steps,)
        self.old_log_probs.clear()

        # Map flat old_log_probs back to (num_ep, max_T) padded form
        old_lp_padded = torch.zeros(num_ep, max_T)
        ptr = 0
        for i, T_i in enumerate(lengths):
            old_lp_padded[i, :T_i] = old_log_probs_flat[ptr : ptr + T_i]
            ptr += T_i

        # ── 4. Bootstrap values with a single forward pass (no grad) ──────
        with torch.no_grad():
            h0 = self.network.init_hidden(batch_size=num_ep)
            (_, _), values_seq, _ = self.network(states, h0)  # (N, T, 1)
            (_, _), next_values_seq, _ = self.network(next_states, h0)  # (N, T, 1)
        values_seq = values_seq.squeeze(-1)  # (N, T)
        next_values_seq = next_values_seq.squeeze(-1)  # (N, T)

        # ── 5. GAE over each episode ───────────────────────────────────────
        advantages_padded = torch.zeros(num_ep, max_T)
        returns_padded = torch.zeros(num_ep, max_T)

        for i, T_i in enumerate(lengths):
            adv = 0.0
            ret = 0.0
            for t in reversed(range(T_i)):
                d = dones[i, t].item()
                r = rewards[i, t].item()
                v = values_seq[i, t].item()
                v_nx = next_values_seq[i, t].item()
                if d:
                    delta = r - v
                    adv = delta
                    ret = 0.0
                else:
                    delta = r + GAMMA * v_nx - v
                    adv = delta + GAMMA * LAMBDA * adv
                advantages_padded[i, t] = adv
                ret = r + GAMMA * ret
                returns_padded[i, t] = ret

        # Normalise advantages and returns over valid steps only
        valid_adv = advantages_padded[mask]
        advantages_padded[mask] = (valid_adv - valid_adv.mean()) / (
            valid_adv.std() + 1e-8
        )

        valid_ret = returns_padded[mask]
        max_abs = valid_ret.abs().max() + 1e-8
        returns_padded = returns_padded / max_abs

        # ── 6. PPO epochs ─────────────────────────────────────────────────
        tot_l, act_l, cri_l, ent_l = [], [], [], []

        episode_indices = list(range(num_ep))

        for _ in range(epochs):
            random.shuffle(episode_indices)

            # Mini-batch over episodes (sequence batches)
            for start in range(0, num_ep, batch_size):
                idx = episode_indices[start : start + batch_size]

                mb_states = states[idx]  # (mb, T, obs)
                mb_actions = actions[idx]  # (mb, T)
                mb_advantages = advantages_padded[idx]  # (mb, T)
                mb_returns = returns_padded[idx]  # (mb, T)
                mb_old_lp = old_lp_padded[idx]  # (mb, T)
                mb_mask = mask[idx]  # (mb, T)

                h0 = self.network.init_hidden(batch_size=len(idx))
                (alpha, beta), values, _ = self.network(mb_states, h0)
                # alpha, beta : (mb, T, 1)
                alpha = alpha.squeeze(-1)  # (mb, T)
                beta = beta.squeeze(-1)
                values = values.squeeze(-1)  # (mb, T)

                dist = self._dist(alpha, beta)

                mb_action_norm = (mb_actions / self.max_speed).clamp(1e-6, 1 - 1e-6)
                new_log_prob = dist.log_prob(mb_action_norm)  # (mb, T)

                ratio = (new_log_prob - mb_old_lp).exp()
                clipped = ratio.clamp(1 - epsilon, 1 + epsilon)

                # Apply mask: zero out padded positions before mean
                adv_masked = mb_advantages * mb_mask
                ratio_masked = ratio * mb_mask
                clipped_masked = clipped * mb_mask

                n_valid = mb_mask.sum().clamp(min=1)

                actor_loss = (
                    -(
                        torch.min(
                            ratio_masked * adv_masked, clipped_masked * adv_masked
                        )
                    ).sum()
                    / n_valid
                )

                critic_loss = (
                    ((mb_returns.detach() - values) ** 2) * mb_mask
                ).sum() / n_valid

                entropy_loss = -(dist.entropy() * mb_mask).sum() / n_valid

                loss = actor_loss + 0.5 * critic_loss + beta_coef * entropy_loss

                tot_l.append(loss.item())
                act_l.append(actor_loss.item())
                cri_l.append(critic_loss.item())
                ent_l.append(entropy_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.gradient_history.append(
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                )
                self.optimizer.step()

        self.tot_loss_history.append(np.mean(tot_l))
        self.actor_loss_history.append(np.mean(act_l))
        self.critic_loss_history.append(np.mean(cri_l))
        self.entropy_loss_history.append(np.mean(ent_l))
        self.reset_hidden()  # next rollout always starts from zero hidden state
