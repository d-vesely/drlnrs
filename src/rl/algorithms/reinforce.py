from collections import deque
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class ActorDiscrete(nn.Module):
    def __init__(self, state_size, item_size, hidden_size):
        super(ActorDiscrete, self).__init__()
        # Input is state and item
        self.fc1 = nn.Linear(state_size + item_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, int(hidden_size / 2))
        # Output is action scores for discrete actions "recommend" and "ignore"
        self.fc5 = nn.Linear(int(hidden_size / 2), 2)

    def forward(self, state, item):
        x = torch.cat((state, item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=-1)


class REINFORCE():
    def __init__(self, device):
        self.device = device
        self.eps = np.finfo(np.float32).eps.item()  # TODO torch

    def act(self, action_probs):
        # print(action_probs)
        m = Categorical(action_probs)
        # print(m)
        action = torch.argmax(action_probs, dim=1)
        # print(action)
        self.log_probs_buffer = m.log_prob(action)
        # print(self.log_probs_buffer)
        return action

    def set_rewards_buffer(self, rewards_buffer):
        self.rewards_buffer = rewards_buffer

    def get_returns(self, episode_length):
        G = 0
        returns = deque(maxlen=episode_length)
        for r in self.rewards_buffer[::-1]:
            G = r + G
            returns.appendleft(G)

        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    def get_loss(self, returns):
        policy_losses = torch.empty(len(returns), device=self.device)
        index = 0
        for log_prob, G in zip(self.log_probs_buffer, returns):
            policy_losses[index] = (-log_prob * G)
            index += 1
        policy_loss = policy_losses.sum()
        return policy_loss

    def reset(self):
        del self.log_probs_buffer
        del self.rewards_buffer
