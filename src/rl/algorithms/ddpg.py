import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, tanh=False):
        super(Actor, self).__init__()
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
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.tanh:
            action = torch.tanh(self.fc5(x))
        else:
            action = self.fc5(x)
        return action


class ActorWithCandidateLSTM(nn.Module):
    def __init__(self, device, state_size, action_size, hidden_size,
                 lstm_hidden_size=2048, lstm_num_layers=1, tanh=False):
        super(ActorWithCandidateLSTM, self).__init__()
        self.device = device
        self.tanh = tanh
        self.LSTM = nn.LSTM(
            action_size + state_size,
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

        state_copy = state.clone().detach()
        state_copy = state_copy.unsqueeze(1)
        state_copy = state_copy.repeat(1, candidates.shape[1], 1)

        lstm_input = torch.cat((candidates, state_copy), dim=-1)
        lstm_output, (hn, cn) = self.LSTM(
            lstm_input,
            (h0.detach(), c0.detach())
        )

        lstm_output = lstm_output[:, -1, :]

        x = torch.cat((state, lstm_output), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.tanh:
            action = torch.tanh(self.fc5(x))
        else:
            action = self.fc5(x)
        return action


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        # Input is state + action
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc4 = nn.Linear(int(hidden_size / 2), 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concat state and action vectors
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


def get_trainee(config_model, device, type):
    if type == "default":
        actor = Actor(**config_model)
        target_actor = Actor(**config_model)
    elif type == "lstm":
        actor = ActorWithCandidateLSTM(**config_model)
        target_actor = ActorWithCandidateLSTM(**config_model)

    critic = Critic(**config_model)
    target_critic = Critic(**config_model)

    actor = actor.to(device)
    target_actor = target_actor.to(device)
    critic = critic.to(device)
    target_critic = target_critic.to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()

    nets = [
        actor, target_actor,
        critic, target_critic
    ]
    target_map = {
        "actor": "target_actor",
        "critic": "target_critic"
    }
    return nets, target_map


def get_evaluatee(config_model, device, type, involved):
    nets = []
    if "a" in involved:
        if type == "default":
            actor = Actor(**config_model)
        elif type == "lstm":
            actor = ActorWithCandidateLSTM(**config_model)

        actor = actor.to(device)
        actor.eval()
        nets.append(actor)
    if "c" in involved:
        critic = Critic(**config_model)

        critic = critic.to(device)
        critic.eval()
        nets.append(critic)

    return nets
