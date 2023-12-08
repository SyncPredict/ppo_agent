import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class ContinuousActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ContinuousActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)  # Добавлен слой Dropout для регуляризации
        )
        self.actor = nn.Linear(128, 2)  # Для среднего и стандартного отклонения
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        actor_output = self.actor(x)

        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = log_std.exp()

        value = self.critic(x)
        return mean, std, value


class ContinuousPPO:
    def __init__(self, num_inputs, num_actions, actor_critic=ContinuousActorCritic, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, epsilon=0.2, c1=0.5, c2=0.01):
        self.actor_critic = actor_critic(num_inputs, num_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2

    def select_action(self, state, training=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mean, std, _ = self.actor_critic(state)
        dist = Normal(mean, std)

        if training:
            action = dist.sample()
        else:
            action = mean  # Детерминированное действие во время тестирования

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy().flatten(), log_prob

    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns

    def update(self, states, actions, log_probs_old, rewards, dones):
        # Предполагая, что log_probs_old, rewards, и dones уже являются тензорами
        log_probs_old_tensor = torch.cat([l.clone().detach().unsqueeze(0) for l in log_probs_old])
        rewards_tensor = torch.cat([r.clone().detach().unsqueeze(0) for r in rewards])
        dones_tensor = torch.cat([d.clone().detach().unsqueeze(0) for d in dones])

        # Если states и actions не являются тензорами, продолжайте использовать torch.stack
        states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        actions_tensor = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions])

        rollouts_dataset = TensorDataset(states_tensor, actions_tensor, log_probs_old_tensor, rewards_tensor,
                                         dones_tensor)
        loader = DataLoader(rollouts_dataset, batch_size=64, shuffle=True)

        for states, actions, log_probs_old, returns, advantages in loader:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            mean, std, values = self.actor_critic(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns, values.squeeze(-1))

            self.optimizer.zero_grad()
            loss = action_loss + self.c1 * value_loss - self.c2 * dist.entropy().mean()
            loss.backward()
            self.optimizer.step()
