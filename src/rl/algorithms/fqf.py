import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_FQF(nn.Module):
    def __init__(self, device, hidden_size, state_item_join_size, n_quantiles):
        super(DQN_FQF, self).__init__()
        self.device = device
        self.phi = nn.Linear(n_quantiles, 256)
        quantiles = torch.arange(
            1, n_quantiles + 1, 1.0, device=self.device)
        self.pi_quantiles = (torch.pi * quantiles)
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        self.n_quantiles = n_quantiles

    def get_embedding(self, state, item):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

    def get_quantiles(self, x, tau):
        cos = torch.cos(self.pi_quantiles * tau.unsqueeze(-1))
        phi_tau = F.relu(self.phi(cos))
        # if len(x.shape) == 3:
        #     phi_tau = phi_tau.unsqueeze(1)

        x = x.unsqueeze(-2)
        x = x * phi_tau

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.transpose(-2, -1)
        return x


class FPN(nn.Module):
    def __init__(self, device, n_quantiles):
        super(FPN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(256, n_quantiles)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
        self.n_quantiles = n_quantiles

    def forward(self, x):
        quantile = F.log_softmax(self.fc1(x), dim=-1)
        probs = quantile.exp()
        if len(quantile.shape) == 3:
            tau_0 = torch.zeros(
                (quantile.shape[0], quantile.shape[1], 1), device=self.device)
        else:
            tau_0 = torch.zeros((len(x), 1), device=self.device)
        tau_1 = torch.cumsum(probs, dim=-1)
        tau = torch.cat((tau_0, tau_1), dim=-1)
        if len(quantile.shape) == 3:
            tau_hat = (tau[:, :, :-1] + tau[:, :, 1:]).detach() / 2
        else:
            tau_hat = (tau[:, :-1] + tau[:, 1:]).detach() / 2
        entropy = (-1) * (quantile * probs).sum(dim=-1, keepdim=True)

        return tau, tau_hat, entropy

    def get_loss(self, FZ_1, FZ_2, tau):
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


def act(quantiles, action=None):
    # q_values = self.get_q_values(quantiles)
    # if action is None:
    #     action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def get_next_action(quantiles, tau):
    q_values = _get_q_values(quantiles, tau)
    q_values = q_values.squeeze(-1)
    action = torch.argmax(q_values, dim=-1)
    return action, quantiles[torch.arange(len(quantiles)), action]


def _get_q_values(quantiles, tau):
    temp = tau[:, :, 1:, None] - tau[:, :, :-1, None]
    temp = temp.squeeze(-1).unsqueeze(-2)
    # if len(quantiles.shape) == 4:
    #     temp = temp.unsqueeze(1)
    q_values = (temp * quantiles).sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_q_values_eval(quantiles, tau):
    temp = tau[:, 1:, None] - tau[:, :-1, None]
    temp = temp.squeeze(-1).unsqueeze(-2)
    # if len(quantiles.shape) == 4:
    #     temp = temp.unsqueeze(1)
    q_values = (temp * quantiles).sum(dim=-1)
    q_values = torch.nan_to_num(q_values, nan=-10000)
    return q_values


def get_trainee(config_model, device):
    type = config_model["type"]
    net_params = config_model["net_params"]
    n_quantiles = net_params["n_quantiles"]

    if type == "default":
        dqn = DQN_FQF(device, **net_params)
        target_dqn = DQN_FQF(device, **net_params)
        fpn = FPN(device, n_quantiles)

    dqn = dqn.to(device)
    target_dqn = target_dqn.to(device)
    fpn = fpn.to(device)

    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    nets = [dqn, target_dqn, fpn]
    target_map = {"dqn": "target_dqn"}
    return nets, target_map


def get_evaluatee(config_model, device):
    type = config_model["type"]
    net_params = config_model["net_params"]
    n_quantiles = net_params["n_quantiles"]

    if type == "default":
        dqn = DQN_FQF(device, **net_params)
        fpn = FPN(device, n_quantiles)

    dqn = dqn.to(device)
    dqn.eval()

    fpn = fpn.to(device)
    fpn.eval()

    nets = [dqn, fpn]
    return nets
