import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    def evaluate(self, state, action):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.policy = ActorCritic(state_dim, action_dim, config.HIDDEN_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LEARNING_RATE)
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

    def act(self, state):
        return self.policy.act(state)

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob.detach())
        self.memory['values'].append(value.detach().squeeze())
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []

    def update(self):
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(np.array(self.memory['actions']))
        old_log_probs = torch.stack(self.memory['log_probs'])
        old_values = torch.stack(self.memory['values'])
        rewards_raw = self.memory['rewards']
        rewards = torch.FloatTensor(np.array(rewards_raw))  # Rewards should already be positive from env
        dones = torch.FloatTensor(self.memory['dones'])
        returns, advantages = self._compute_returns_advantages(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.config.PPO_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPSILON, 1 + self.config.CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                loss = actor_loss + self.config.VALUE_COEF * critic_loss - self.config.ENTROPY_COEF * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.MAX_GRAD_NORM if hasattr(self.config, "MAX_GRAD_NORM") else 0.5)
                self.optimizer.step()
        self.clear_memory()

    def _compute_returns_advantages(self, rewards, values, dones):
        rewards = rewards.clone().detach()
        values = values.clone().detach()
        dones = dones.clone().detach()
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae_lam = 0
        gamma = self.config.GAMMA
        lam = 0.95
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        returns = advantages + values
        return returns, advantages

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
