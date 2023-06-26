import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorTD3(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, tanh=False):
        super(ActorTD3, self).__init__()
        self.tanh = tanh
        # Input is state
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, int(hidden_size / 2))
        # Output is action
        self.fc5 = nn.Linear(int(hidden_size / 2), action_size)
        self.fc5.weight.data.uniform_(-4e-1, 4e-1)
        self.fc5.bias.data.uniform_(-4e-1, 4e-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_embedding = F.relu(self.fc3(x))
        x = state_embedding.clone()
        x = F.relu(self.fc4(x))
        if self.tanh:
            action = torch.tanh(self.fc5(x))
        else:
            action = self.fc5(x)
        return action, state_embedding


class ActorTD3_LSTM(nn.Module):
    def __init__(self, device, state_size, action_size, hidden_size,
                 lstm_hidden_size=2048, lstm_num_layers=2, tanh=False):
        super(ActorTD3_LSTM, self).__init__()
        self.device = device
        self.tanh = tanh
        self.LSTM = nn.LSTM(
            action_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True
        )
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.fc1 = nn.Linear(state_size + lstm_hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, int(hidden_size / 2))
        # Output is action
        self.fc5 = nn.Linear(int(hidden_size / 2), action_size)
        self.fc5.weight.data.uniform_(-4e-1, 4e-1)
        self.fc5.bias.data.uniform_(-4e-1, 4e-1)

    def forward(self, state, candidates):
        if len(candidates.shape) == 2:
            candidates = candidates.unsqueeze(0)
            state = state.unsqueeze(0)
        h0 = torch.zeros(
            self.lstm_num_layers,
            candidates.shape[0],
            self.lstm_hidden_size
        ).requires_grad_().to(self.device)
        c0 = torch.zeros(
            self.lstm_num_layers,
            candidates.shape[0],
            self.lstm_hidden_size
        ).requires_grad_().to(self.device)

        lstm_output, (hn, cn) = self.LSTM(
            candidates,
            (h0.detach(), c0.detach())
        )

        lstm_output = lstm_output[:, -1, :]

        x = torch.cat((state, lstm_output), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_embedding = F.relu(self.fc3(x))
        x = state_embedding.clone()
        x = F.relu(self.fc4(x))
        if self.tanh:
            action = torch.tanh(self.fc5(x))
        else:
            action = self.fc5(x)
        return action, state_embedding


class CriticTD3(nn.Module):
    def __init__(self, state_embedding_size, action_size, hidden_size):
        super(CriticTD3, self).__init__()
        # Input is state + action
        self.fc1 = nn.Linear(state_embedding_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), 256)
        # Output is single q-value
        self.fc4 = nn.Linear(256, 1)

    def forward(self, state_embedding, action):
        # Concat state and action vectors
        x = torch.cat((state_embedding, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


def get_trainee(config_model, device, type):
    if type == "default":
        actor = ActorTD3(**config_model)
        target_actor = ActorTD3(**config_model)
    elif type == "lstm":
        actor = ActorTD3_LSTM(**config_model)
        target_actor = ActorTD3_LSTM(**config_model)

    critic_1 = CriticTD3(**config_model)
    critic_2 = CriticTD3(**config_model)
    target_critic_1 = CriticTD3(**config_model)
    target_critic_2 = CriticTD3(**config_model)

    actor = actor.to(device)
    target_actor = target_actor.to(device)
    critic_1 = critic_1.to(device)
    critic_2 = critic_2.to(device)
    target_critic_1 = target_critic_1.to(device)
    target_critic_2 = target_critic_2.to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_1.eval()
    target_critic_2.load_state_dict(critic_2.state_dict())
    target_critic_2.eval()

    nets = [
        actor, target_actor,
        critic_1, target_critic_1,
        critic_2, target_critic_2
    ]
    target_map = {
        "actor": "target_actor",
        "critic_1": "target_critic_1",
        "critic_2": "target_critic_2",
    }
    return nets, target_map


def get_evaluatee(config_model, device, type):
    if type == "default":
        actor = ActorTD3(**config_model)
    elif type == "lstm":
        actor = ActorTD3_LSTM(**config_model)

    actor = actor.to(device)
    actor.eval()

    nets = [actor]
    return nets
