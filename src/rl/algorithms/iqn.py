import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_IQN(nn.Module):
    """Z-network for IQN"""

    def __init__(self, device, hidden_size, state_item_join_size, n_quantiles):
        super(DQN_IQN, self).__init__()
        self.device = device
        # Phi-network (single layer)
        self.phi = nn.Linear(n_quantiles, 256)
        self.n_quantiles = n_quantiles
        quantiles = torch.arange(
            1, n_quantiles + 1, 1.0, device=self.device
        )
        self.pi_quantiles = (torch.pi * quantiles)

        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        self.fc5 = nn.Linear(256, 256)
        # Output is a single quantile
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, item):
        # Mask padded candidates
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask

        # Sample n_quantiles from a uniform distribution
        tau = torch.rand(
            (state.shape[0], self.n_quantiles, 1),
            device=self.device
        )
        # Expand taus and send through phi-network
        cos = torch.cos(self.pi_quantiles * tau)
        phi_tau = F.relu(self.phi(cos))
        if len(state.shape) == 3:
            phi_tau = phi_tau.unsqueeze(1)

        # Send state + item through network
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Element-wise product of hidden state and output of phi
        x = x.unsqueeze(-2)
        x = x * phi_tau

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.transpose(-2, -1)

        # Return quantile and sampled taus
        return x, tau


def get_next_action(quantiles, n_quantiles):
    """Get best action"""
    q_values = get_q_values(quantiles, n_quantiles)
    q_values = q_values.squeeze(-1)
    action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def get_q_values(quantiles, n_quantiles):
    """Compute q-values for given quantiles"""
    # Analogous to QR-DQN
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
        dqn = DQN_IQN(device, **net_params)
        target_dqn = DQN_IQN(device, **net_params)

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
        dqn = DQN_IQN(device, **net_params)

    dqn = dqn.to(device)
    dqn.eval()

    nets = [dqn]
    return nets
