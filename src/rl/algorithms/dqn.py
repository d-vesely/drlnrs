import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, hidden_size, state_item_join_size):
        """Initialize DQN layers

        Input is the state s and an action a.
        Output is a single q-value Q(s, a).

        Keyword Arguments:
            state_size -- size of the state representation vector
            item_size -- size of the action representation vector
            hidden_size -- size of hidden layer
        """
        super(DQN, self).__init__()
        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, item):
        # item = (item - item.mean(dim=-1).unsqueeze(-1)) / \
        #     item.std(dim=-1).unsqueeze(-1)
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


class DQNDueling(nn.Module):
    """Deep Q-Network"""

    def __init__(self, hidden_size, state_item_join_size):
        """Initialize DQN layers

        Input is the state s and an action a.
        Output is a single q-value Q(s, a).

        Keyword Arguments:
            state_size -- size of the state representation vector (default: {768})
            action_size -- size of the action representation vector (default: {768})
            hidden_size -- size of hidden layer (default: {2048})
        """
        super(DQNDueling, self).__init__()
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 256)
        self.value_fc3 = nn.Linear(256, 1)

        self.advantage_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_fc2 = nn.Linear(hidden_size // 2, 256)
        self.advantage_fc3 = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, state, item):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask
        x = torch.cat((state, masked_item), dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        value_x = F.leaky_relu(self.value_fc1(x))
        value_x = F.leaky_relu(self.value_fc2(value_x))
        value_x = F.leaky_relu(self.value_fc3(value_x))

        advantage_x = F.leaky_relu(self.advantage_fc1(x))
        advantage_x = F.leaky_relu(self.advantage_fc2(advantage_x))
        advantage_x = F.leaky_relu(self.advantage_fc3(advantage_x))

        return value_x, advantage_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


class DQNAttention(nn.Module):
    """Deep Q-Network"""

    def __init__(self, hidden_size, state_item_join_size, item_size, mode):
        """Initialize DQN layers

        Input is the state s and an action a.
        Output is a single q-value Q(s, a).

        Keyword Arguments:
            state_size -- size of the state representation vector
            item_size -- size of the action representation vector
            hidden_size -- size of hidden layer
        """
        super(DQNAttention, self).__init__()

        self.att = nn.MultiheadAttention(item_size, 8, batch_first=True)
        if mode == "additive":
            self.W = nn.Linear(768, 256)
            self.V = nn.Linear(256, 1)
        self.mode = mode
        # Input is state + action
        self.fc1 = nn.Linear(state_item_join_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, item, test=False):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask

        if not test:
            repeat_state = False
            if len(state.shape) == 4:
                state = state[:, 0]
                repeat_state = True
        else:
            repeat_state = False
            if len(state.shape) == 3:
                state = state[0].unsqueeze(0)
                repeat_state = True

        padding_mask = (state == 0).all(dim=2)
        all_true_rows = (padding_mask.sum(dim=1) == padding_mask.shape[1])
        padding_mask[all_true_rows, 0] = False
        attn_output, _ = self.att(
            state, state, state,
            key_padding_mask=padding_mask
        )

        if self.mode == "additive":
            scores = self.V(torch.tanh(self.W(attn_output))).squeeze(2)
            att_weights = torch.softmax(scores, dim=1)
            state = torch.bmm(
                att_weights.unsqueeze(1), state).squeeze(1)
        elif self.mode == "mean":
            state = torch.mean(attn_output, dim=1)

        if not test:
            if repeat_state:
                state = state.unsqueeze(
                    1).repeat(1, item.shape[1], 1)
        else:
            if repeat_state:
                state = state.repeat(len(item), 1)

        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


class DQNLSTM(nn.Module):
    """Deep Q-Network"""

    def __init__(self, device, news_emb_layers, norm, item_size, hidden_size):
        """Initialize DQN layers

        Input is the state s and an action a.
        Output is a single q-value Q(s, a).

        Keyword Arguments:
            state_size -- size of the state representation vector
            item_size -- size of the action representation vector
            hidden_size -- size of hidden layer
        """
        super(DQNLSTM, self).__init__()
        self.device = device
        # News embedding layers

        self.lstm_hidden_size = 768
        self.lstm_num_layers = 1

        self.LSTM = nn.LSTM(
            item_size,
            self.lstm_hidden_size,
            self.lstm_num_layers,
            batch_first=True,
        )

        # Input is state + action
        self.fc1 = nn.Linear(
            item_size + self.lstm_hidden_size,
            hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, item, test=False):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask

        if not test:
            repeat_state = False
            if len(state.shape) == 4:
                state = state[:, 0]
                repeat_state = True
        else:
            repeat_state = False
            if len(state.shape) == 3:
                state = state[0].unsqueeze(0)
                repeat_state = True

        h0 = torch.zeros(
            self.lstm_num_layers,
            state.shape[0],
            self.lstm_hidden_size
        ).requires_grad_().to(self.device)
        c0 = torch.zeros(
            self.lstm_num_layers,
            state.shape[0],
            self.lstm_hidden_size
        ).requires_grad_().to(self.device)

        lstm_output, (hn, cn) = self.LSTM(
            state,
            (h0.detach(), c0.detach())
        )
        lstm_output = lstm_output[:, -1, :]

        if not test:
            if repeat_state:
                lstm_output = lstm_output.unsqueeze(
                    1).repeat(1, item.shape[1], 1)
        else:
            if repeat_state:
                lstm_output = lstm_output.repeat(len(item), 1)

        if lstm_output.shape[0] != item.shape[0]:
            lstm_output = lstm_output.view(
                item.shape[0],
                item.shape[1],
                -1
            )
        x = torch.cat((lstm_output, masked_item), dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


class DQNGRU(nn.Module):
    """Deep Q-Network"""

    def __init__(self, device, item_size, hidden_size, mode):
        """Initialize DQN layers

        Input is the state s and an action a.
        Output is a single q-value Q(s, a).

        Keyword Arguments:
            state_size -- size of the state representation vector
            item_size -- size of the action representation vector
            hidden_size -- size of hidden layer
        """
        super(DQNGRU, self).__init__()
        self.device = device

        self.gru_hidden_size = 768
        self.gru_num_layers = 1

        self.GRU = nn.GRU(
            item_size,
            self.gru_hidden_size,
            self.gru_num_layers,
            batch_first=True,
            bidirectional=True
        )
        if mode == "additive":
            self.W = nn.Linear(1536, 256)
            self.V = nn.Linear(256, 1)
        self.mode = mode

        # Input is state + action
        self.fc1 = nn.Linear(
            (item_size * 2) + self.gru_hidden_size,
            hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, item, test=False):
        mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
        masked_item = item * mask

        if not test:
            repeat_state = False
            if len(state.shape) == 4:
                state = state[:, 0]
                repeat_state = True
        else:
            repeat_state = False
            if len(state.shape) == 3:
                state = state[0].unsqueeze(0)
                repeat_state = True

        padding_mask = (state == 0).all(dim=2)
        # Count the number of rows that are all zeros for each element in the batch
        lengths = (state.shape[1] - padding_mask.sum(dim=1)).cpu()
        lengths = torch.where(lengths == 0, torch.tensor(1), lengths)
        state_packed = pack_padded_sequence(
            state, lengths, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(
            self.gru_num_layers * 2,
            state.shape[0],
            self.gru_hidden_size
        ).requires_grad_().to(self.device)

        gru_output, _ = self.GRU(
            state_packed,
            h0
        )
        gru_output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_output, batch_first=True)

        if self.mode == "additive":
            scores = self.V(torch.tanh(self.W(gru_output_unpacked))).squeeze(2)
            att_weights = torch.softmax(scores, dim=1)
            state = torch.bmm(
                att_weights.unsqueeze(1), gru_output_unpacked).squeeze(1)
        elif self.mode == "mean":
            state = gru_output_unpacked.mean(dim=1)
        elif self.mode == "last":
            state = gru_output_unpacked[:, -1, :]

        if not test:
            if repeat_state:
                state = state.unsqueeze(
                    1).repeat(1, item.shape[1], 1)
        else:
            if repeat_state:
                state = state.repeat(len(item), 1)

        if state.shape[0] != item.shape[0]:
            state = state.view(
                item.shape[0],
                item.shape[1],
                -1
            )
        x = torch.cat((state, masked_item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


def get_trainee(config_model, device):
    type = config_model["type"]
    net_params = config_model["net_params"]

    if type == "default":
        dqn = DQN(**net_params)
        target_dqn = DQN(**net_params)
    elif type == "dueling":
        dqn = DQNDueling(**net_params)
        target_dqn = DQNDueling(**net_params)
    elif type == "att":
        dqn = DQNAttention(**net_params)
        target_dqn = DQNAttention(**net_params)
    elif type == "lstm":
        dqn = DQNLSTM(device, **net_params)
        target_dqn = DQNLSTM(device, **net_params)
    elif type == "gru":
        dqn = DQNGRU(device, **net_params)
        target_dqn = DQNGRU(device, **net_params)

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
        dqn = DQN(**net_params)
    elif type == "dueling":
        dqn = DQNDueling(**net_params)
    elif type == "att":
        dqn = DQNAttention(**net_params)
    elif type == "lstm":
        dqn = DQNLSTM(device, **net_params)
    elif type == "gru":
        dqn = DQNGRU(device, **net_params)

    dqn = dqn.to(device)
    dqn.eval()

    nets = [dqn]
    return nets
