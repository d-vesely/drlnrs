import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_IQN(nn.Module):
    def __init__(self, device, hidden_size, state_item_join_size, n_quantiles):
        super(DQN_IQN, self).__init__()
        self.device = device
        self.phi = nn.Linear(n_quantiles, 256)
        quantiles = torch.arange(
            1, n_quantiles + 1, 1.0, device=self.device)
        self.pi_quantiles = (torch.pi * quantiles)
        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        self.n_quantiles = n_quantiles

    def forward(self, state, item):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask

        tau = torch.rand(
            (state.shape[0], self.n_quantiles, 1),
            device=self.device
        )
        cos = torch.cos(self.pi_quantiles * tau)
        phi_tau = F.relu(self.phi(cos))
        if len(state.shape) == 3:
            phi_tau = phi_tau.unsqueeze(1)

        x = torch.cat((state, masked_item), dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = x.unsqueeze(-2)
        x = x * phi_tau

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.transpose(-2, -1)
        return x, tau


def act(quantiles, action=None):
    q_values = get_q_values(quantiles)
    if action is None:
        action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def get_next_action(quantiles, n_quantiles):
    q_values = get_q_values(quantiles, n_quantiles)
    q_values = q_values.squeeze(-1)
    action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def get_q_values(quantiles, n_quantiles):
    q_values = quantiles / n_quantiles
    q_values = q_values.sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_trainee(config_model, device):
    type = config_model["type"]
    net_params = config_model["net_params"]

    if type == "default":
        dqn = DQN_IQN(device, **net_params)
        target_dqn = DQN_IQN(device, **net_params)

    dqn = dqn.to(device)
    target_dqn = target_dqn.to(device)

    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    nets = [dqn, target_dqn]
    target_map = {"dqn": "target_dqn"}
    return nets, target_map


def get_evaluatee(config_model, device):
    type = config_model["type"]
    net_params = config_model["net_params"]

    if type == "default":
        dqn = DQN_IQN(device, **net_params)

    dqn = dqn.to(device)
    dqn.eval()

    nets = [dqn]
    return nets
