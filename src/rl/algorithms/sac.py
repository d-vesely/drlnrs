import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class ActorSAC(nn.Module):
    def __init__(self, state_size, item_size, hidden_size):
        super(ActorSAC, self).__init__()
        # Input is state and item
        self.fc1 = nn.Linear(state_size + item_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, int(hidden_size / 2))
        # Output is action scores for discrete actions "recommend" and "ignore"
        self.fc5 = nn.Linear(int(hidden_size / 2), 2)

    def forward(self, state, item):
        x = torch.cat((state, item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SoftCritic(nn.Module):
    def __init__(self, state_size, item_size, hidden_size, batched=True):
        super(SoftCritic, self).__init__()
        # Input is state + action
        self.fc1 = nn.Linear(state_size + item_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc4 = nn.Linear(int(hidden_size / 2), 256)
        # Output is single q-value
        self.fc5 = nn.Linear(256, 2)

        self.batched = batched

    def forward(self, state, item):
        if self.batched:
            mask = torch.all(torch.isfinite(item), dim=-1).unsqueeze(-1)
            masked_item = item * mask
            # Concat state and action vectors
            x = torch.cat((state, masked_item), dim=-1)
        else:
            x = torch.cat((state, item), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value


def get_action_probs(action_logits):
    dist = Categorical(logits=action_logits)
    action_probs = dist.probs
    log_probs = F.log_softmax(action_logits, dim=1)
    return action_probs, log_probs


def get_trainee(config_model, device, type="default"):
    if type == "default":
        actor = ActorSAC(**config_model)
        soft_critic_1 = SoftCritic(**config_model)
        soft_critic_2 = SoftCritic(**config_model)
        target_soft_critic_1 = SoftCritic(**config_model)
        target_soft_critic_2 = SoftCritic(**config_model)

    actor = actor.to(device)
    soft_critic_1 = soft_critic_1.to(device)
    soft_critic_2 = soft_critic_2.to(device)
    target_soft_critic_1 = target_soft_critic_1.to(device)
    target_soft_critic_2 = target_soft_critic_2.to(device)

    target_soft_critic_1.load_state_dict(soft_critic_1.state_dict())
    target_soft_critic_1.eval()
    target_soft_critic_2.load_state_dict(soft_critic_2.state_dict())
    target_soft_critic_2.eval()

    nets = [
        actor,
        soft_critic_1, target_soft_critic_1,
        soft_critic_2, target_soft_critic_2
    ]
    target_map = {
        "soft_critic_1": "target_soft_critic_1",
        "soft_critic_2": "target_soft_critic_2"
    }
    return nets, target_map


def get_evaluatee(config_model, device, type):
    actor = ActorSAC(**config_model)
    actor = actor.to(device)
    actor.eval()
    nets = [actor]
    return nets
