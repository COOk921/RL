import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        :param state_dim: 输入维度
        :param action_dim: 输出维度
        """
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.view(x.size(0),-1)        #[B, 25]
        x = torch.tanh(self.fc(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.softmax(self.logits_head(x), dim=-1)
        dist = torch.distributions.Categorical(prob)
        # logits = self.logits_head(x)
        # dist = torch.distributions.Categorical(logits=logits)
        return dist


# Critic Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)     

    def forward(self, state): #[B, 5, 5]
        
        state = state.view(state.size(0), -1) #[B, 25]
        x = torch.tanh(self.fc(state))
        x = torch.tanh(self.fc2(x))
        value = self.fc_value(x)
        return value


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim,device ,lr_actor=3e-5, lr_critic=1e-5, gamma=0.99,
                 clip_eps=0.2, epochs=10, batch_size=64):
        self.device = device
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.value.parameters(), lr=lr_critic)

        self.gamma = gamma      # 折扣因子
        self.clip_eps = clip_eps # PPO clip参数
        self.epochs = epochs  # PPO循环次数
        self.batch_size = batch_size # PPO batch_size

        self.buffer_states = []  # 存储state
        self.buffer_actions = []  # 存储action
        self.buffer_log_probs = [] # 存储log_prob
        self.buffer_rewards = []  # 存储reward
        self.buffer_dones = []   # 存储done

    def select_action(self, state):
        # state = state.view(1, -1)
        state = state.unsqueeze(0) 
        dist = self.policy(state)
        action = dist.sample()   
        log_prob = dist.log_prob(action)
        
        return action, log_prob

    def store_transition(self, state, action, log_prob, reward, done):
        self.buffer_states.append(state.cpu().numpy())
        self.buffer_actions.append(action)
        self.buffer_log_probs.append(log_prob.cpu().detach().numpy() )
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
    
    def update(self):
      
        buffer_states = torch.FloatTensor(np.array(self.buffer_states)).to(self.device)
        buffer_actions = torch.FloatTensor(np.array(self.buffer_actions)).to(self.device)
        buffer_log_probs = torch.FloatTensor(np.array(self.buffer_log_probs)).to(self.device)
        buffer_rewards = np.array(self.buffer_rewards)
        buffer_dones = np.array(self.buffer_dones)

        returns = []
        discounted_return = 0
        for r, d in zip(reversed(buffer_rewards), reversed(buffer_dones)):
            if d:
                discounted_return = 0
            discounted_return = r + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.FloatTensor(returns).to(self.device)
       
        advantage = returns - self.value(buffer_states).squeeze(-1).detach()
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6) # 归一化

        for _ in range(self.epochs):
            indices = np.arange(len(buffer_states))
            np.random.shuffle(indices)
            for start in range(0, len(buffer_states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = buffer_states[batch_indices]
                batch_actions = buffer_actions[batch_indices]
                batch_log_probs = buffer_log_probs[batch_indices]
                batch_advantage = advantage[batch_indices]
                batch_returns = returns[batch_indices]

                dist = self.policy(batch_states)
                batch_action_log_probs = dist.log_prob(batch_actions).sum(-1)
                ratio = torch.exp(batch_action_log_probs - batch_log_probs)

                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantage
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                critic_loss = ((self.value(batch_states).squeeze(-1) - batch_returns) ** 2).mean()
                
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_dones = []