import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_QR(nn.Module):
    """Z-network for QR-DQN"""

    def __init__(self, hidden_size, state_item_join_size, n_quantiles):
        super(DQN_QR, self).__init__()
        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is number of quantiles
        self.fc5 = nn.Linear(256, n_quantiles)
        self.n_quantiles = n_quantiles

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
        quantiles = self.fc5(x)
        return quantiles


def get_tau(n_quantiles, device):
    """Get quantile midpoints for given amount"""
    # Create linspace from 0 to 1
    temp = torch.linspace(0.0, 1.0, n_quantiles + 1, device=device)
    # Get quantile midpoints
    tau = (temp[:-1] + temp[1:]) / 2
    return tau


def get_next_action(quantiles, n_quantiles):
    """Get best action"""
    q_values = get_q_values(quantiles, n_quantiles)
    action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def get_q_values(quantiles, n_quantiles):
    """Compute q-values for given quantiles"""
    # q-values are sum over all (quantiles * 1/n_quantiles)
    q_values = quantiles / n_quantiles
    q_values = q_values.sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_trainee(config_model, device):
    """Prepare networks for training"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init online and target nets
    if type == "default":
        dqn = DQN_QR(**net_params)
        target_dqn = DQN_QR(**net_params)

    # Send nets to device
    dqn = dqn.to(device)
    target_dqn = target_dqn.to(device)

    # Copy online net to target net
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    # Map online and target net to each other
    nets = [dqn, target_dqn]
    target_map = {"dqn": "target_dqn"}
    return nets, target_map


def get_evaluatee(config_model, device):
    """Prepare networks for evaluation"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init net
    if type == "default":
        dqn = DQN_QR(**net_params)

    dqn = dqn.to(device)
    dqn.eval()

    nets = [dqn]
    return nets
