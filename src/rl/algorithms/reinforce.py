from collections import deque
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class ActorDiscrete(nn.Module):
    """Discrete Actor for REINFORCE"""

    def __init__(self, hidden_size, state_item_join_size):
        super(ActorDiscrete, self).__init__()
        # Input is state and item
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is action scores for discrete actions "recommend" and "ignore"
        self.fc5 = nn.Linear(256, 2)

    def forward(self, state, item):
        # Send state + item through network
        x = torch.cat((state, item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # Convert logits to probabilities
        return F.softmax(x, dim=-1)


def get_trainee(config_model, device):
    """Prepare networks for training"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init network
    if type == "default":
        actor = ActorDiscrete(**net_params)

    # Send network to device
    actor = actor.to(device)

    nets = [actor]
    target_map = {"actor": None}
    return nets, target_map


def get_evaluatee(config_model, device):
    """Prepare networks for evaluation"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init net
    if type == "default":
        actor = ActorDiscrete(**net_params)

    actor = actor.to(device)
    actor.eval()

    nets = [actor]
    return nets


class REINFORCE():
    """Wrapper for REINFORCE methods"""

    def __init__(self, device):
        """Initialize wrapper"""
        self.device = device
        self.eps = torch.finfo(torch.float32).eps

    def act(self, action_probs):
        """Select action from action probabilities"""
        m = Categorical(action_probs)
        action = torch.argmax(action_probs, dim=1)
        # Save log-probabilities
        self.log_probs_buffer = m.log_prob(action)
        return action

    def set_rewards_buffer(self, rewards_buffer):
        """Set rewards buffer"""
        self.rewards_buffer = rewards_buffer

    def get_returns(self, episode_length, gamma):
        """Compute returns from rewards buffer"""
        G = 0
        returns = deque(maxlen=episode_length)
        # Iterate from last to first reward and incrementally update return
        for r in self.rewards_buffer[::-1]:
            G = r + (gamma * G)
            returns.appendleft(G)

        # Normalize returns
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    def get_loss(self, returns):
        """Compute loss from given returns"""
        policy_losses = torch.empty(len(returns), device=self.device)
        index = 0
        # Multiply each return with each negative log-prob
        for log_prob, G in zip(self.log_probs_buffer, returns):
            policy_losses[index] = (-log_prob * G)
            index += 1
        # Sum over all losses
        policy_loss = policy_losses.sum()
        return policy_loss

    def reset(self):
        """Reset REINFORCE wrapper"""
        # Delete buffers
        del self.log_probs_buffer
        del self.rewards_buffer
