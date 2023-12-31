import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor-network for DDPG"""

    def __init__(self, state_size, item_size, hidden_size, tanh=False):
        super(Actor, self).__init__()
        self.tanh = tanh
        # Input is state
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        # Output is action
        self.fc5 = nn.Linear(hidden_size // 2, item_size)
        self.fc5.weight.data.uniform_(-4e-1, 4e-1)
        self.fc5.bias.data.uniform_(-4e-1, 4e-1)

    def forward(self, state):
        # Send state through network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.tanh:
            action = torch.tanh(self.fc5(x))
        else:
            action = self.fc5(x)
        return action


class Critic(nn.Module):
    """Critic-network for DDPG"""

    def __init__(self, state_size, item_size, hidden_size):
        super(Critic, self).__init__()
        # Input is state + action
        state_item_join_size = state_size + item_size
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, item):
        # Mask padded candidates
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask
        # Send state + item through network
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


def get_trainee(config_model, device):
    """Prepare networks for training"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init online and target nets
    if type == "default":
        actor = Actor(**net_params)
        target_actor = Actor(**net_params)
        critic = Critic(**net_params)
        target_critic = Critic(**net_params)

    # Send nets to device
    actor = actor.to(device)
    target_actor = target_actor.to(device)
    critic = critic.to(device)
    target_critic = target_critic.to(device)

    # Copy online net to target net
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()

    # Map online and target net to each other
    nets = [
        actor, target_actor,
        critic, target_critic
    ]
    target_map = {
        "actor": "target_actor",
        "critic": "target_critic"
    }
    return nets, target_map


def get_evaluatee(config_model, device, involved):
    """Prepare networks for evaluation"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init nets depending on whether evaluation uses just the actor
    # or also the critic
    nets = []
    if "a" in involved:
        if type == "default":
            actor = Actor(**net_params)
        actor = actor.to(device)
        actor.eval()
        nets.append(actor)
    if "c" in involved:
        critic = Critic(**net_params)
        critic = critic.to(device)
        critic.eval()
        nets.append(critic)

    return nets
