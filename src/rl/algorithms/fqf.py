import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_FQF(nn.Module):
    """Z-network for FQF"""

    def __init__(self, device, hidden_size, state_item_join_size, n_quantiles):
        super(DQN_FQF, self).__init__()
        self.device = device
        # Analogous to IQN
        # Phi-network (single layer)
        self.phi = nn.Linear(n_quantiles, 256)
        self.n_quantiles = n_quantiles
        quantiles = torch.arange(
            1, n_quantiles + 1, 1.0, device=self.device)
        self.pi_quantiles = (torch.pi * quantiles)

        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        self.fc5 = nn.Linear(256, 256)
        # Output is a single quantile
        self.fc6 = nn.Linear(256, 1)

    def get_embedding(self, state, item):
        # Mask padded candidates
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask
        # Send state + item through network
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def get_quantiles(self, x, tau):
        # Expand taus produced by FPN and send through phi-network
        cos = torch.cos(self.pi_quantiles * tau.unsqueeze(-1))
        phi_tau = F.relu(self.phi(cos))
        # if len(x.shape) == 3:
        #     phi_tau = phi_tau.unsqueeze(1)

        # Element-wise product of hidden state and output of phi
        x = x.unsqueeze(-2)
        x = x * phi_tau

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.transpose(-2, -1)
        return x


class FPN(nn.Module):
    """Fraction Proposal Network for FQF"""

    def __init__(self, device, n_quantiles):
        super(FPN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(256, n_quantiles)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
        self.n_quantiles = n_quantiles

    def forward(self, x):
        quantile = F.log_softmax(self.fc1(x), dim=-1)

        # Ensure that taus are sorted and tau_0 = 0 and tau_1 = 1
        probs = quantile.exp()
        if len(quantile.shape) == 3:
            tau_0 = torch.zeros(
                (quantile.shape[0], quantile.shape[1], 1), device=self.device)
        else:
            tau_0 = torch.zeros((len(x), 1), device=self.device)
        tau_1 = torch.cumsum(probs, dim=-1)
        tau = torch.cat((tau_0, tau_1), dim=-1)

        # Get quantile midpoints
        if len(quantile.shape) == 3:
            tau_hat = (tau[:, :, :-1] + tau[:, :, 1:]).detach() / 2
        else:
            tau_hat = (tau[:, :-1] + tau[:, 1:]).detach() / 2

        # Entropy-term from the paper
        entropy = (-1) * (quantile * probs).sum(dim=-1, keepdim=True)

        return tau, tau_hat, entropy

    def get_loss(self, FZ_1, FZ_2, tau):
        """Compute FPN loss"""
        gradients1 = FZ_2 - FZ_1[:, :-1]
        gradients2 = FZ_2 - FZ_1[:, 1:]
        flag_1 = FZ_2 > torch.cat([FZ_1[:, :1], FZ_2[:, :-1]], dim=1)
        flag_2 = FZ_2 < torch.cat([FZ_2[:, 1:], FZ_1[:, -1:]], dim=1)
        gradients = (torch.where(flag_1, gradients1, -gradients1) +
                     torch.where(flag_2, gradients2, -gradients2))
        gradients = gradients.view(tau.shape[0], self.n_quantiles-1)
        assert not gradients.requires_grad
        loss = (gradients * tau[:, 1:-1]).sum(dim=-1).mean()
        return loss


def get_next_action(quantiles, tau):
    """Get best action"""
    q_values = _get_q_values(quantiles, tau)
    q_values = q_values.squeeze(-1)
    action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def _get_q_values(quantiles, tau):
    """Compute q-values for given quantiles and tau"""
    # Get distances between taus
    temp = tau[:, :, 1:, None] - tau[:, :, :-1, None]
    temp = temp.squeeze(-1).unsqueeze(-2)
    # q-values is sum over all (distances * quantiles)
    q_values = (temp * quantiles).sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_q_values_eval(quantiles, tau):
    """Compute q-values for given quantiles and tau"""
    # Analogous to above, but unbatched
    temp = tau[:, 1:, None] - tau[:, :-1, None]
    temp = temp.squeeze(-1).unsqueeze(-2)
    q_values = (temp * quantiles).sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_trainee(config_model, device):
    """Prepare networks for training"""
    type = config_model["type"]
    net_params = config_model["net_params"]
    n_quantiles = net_params["n_quantiles"]

    # Init online and target nets
    if type == "default":
        dqn = DQN_FQF(device, **net_params)
        target_dqn = DQN_FQF(device, **net_params)
        fpn = FPN(device, n_quantiles)

    # Send nets to device
    dqn = dqn.to(device)
    target_dqn = target_dqn.to(device)
    fpn = fpn.to(device)

    # Copy online net to target net
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    # Map online and target net to each other
    nets = [dqn, target_dqn, fpn]
    target_map = {"dqn": "target_dqn"}
    return nets, target_map


def get_evaluatee(config_model, device):
    """Prepare networks for evaluation"""
    type = config_model["type"]
    net_params = config_model["net_params"]
    n_quantiles = net_params["n_quantiles"]

    # Init nets
    if type == "default":
        dqn = DQN_FQF(device, **net_params)
        fpn = FPN(device, n_quantiles)

    dqn = dqn.to(device)
    dqn.eval()

    fpn = fpn.to(device)
    fpn.eval()

    nets = [dqn, fpn]
    return nets
