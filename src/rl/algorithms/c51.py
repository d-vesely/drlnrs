import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_C51(nn.Module):
    """Z-network for C51"""

    def __init__(self, hidden_size, state_item_join_size,
                 v, n_atoms):
        super(DQN_C51, self).__init__()
        # Register supports in own buffer
        self.register_buffer("supports", torch.linspace(-v, v, n_atoms))
        # Input is state + item
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is number of atoms
        self.fc5 = nn.Linear(256, n_atoms)
        self.n_atoms = n_atoms

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
        x = self.fc5(x)
        x = x.unsqueeze(1)
        # Softmax over atom logits
        pmfs = F.softmax(x, dim=-1)
        return pmfs


def get_next_action(pmfs, supports):
    """Get best action"""
    q_values = get_q_values(pmfs, supports)
    q_values = q_values.squeeze(-1)
    action = torch.argmax(q_values, dim=-1)
    return action, pmfs[torch.arange(len(pmfs)), action, 0]


def get_q_values(pmfs, supports):
    """Compute q-values for given PMFs and supports"""
    # q-values are sum over all (probabilities * support)
    q_values = pmfs * supports
    q_values = q_values.sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_trainee(config_model, device):
    """Prepare networks for training"""
    type = config_model["type"]
    net_params = config_model["net_params"]

    # Init online and target nets
    if type == "default":
        dqn = DQN_C51(**net_params)
        target_dqn = DQN_C51(**net_params)

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
        dqn = DQN_C51(**net_params)

    dqn = dqn.to(device)
    dqn.eval()

    nets = [dqn]
    return nets
