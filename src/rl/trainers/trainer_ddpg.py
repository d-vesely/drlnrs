import torch
import torch.nn as nn
import torch.optim as optim

from ..algorithms.ddpg import get_trainee
from .trainer_base import _TrainerBase


class TrainerDDPG(_TrainerBase):
    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params, rm_episodic_path=""):
        super().__init__(
            model_name,
            device,
            pos_rm_path,
            neg_rm_path,
            encoder_params,
            learning_params,
            model_params,
            rm_episodic_path
        )

    def set_trainee(self, type):
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
            type
        )
        self.actor, self.target_actor, \
            self.critic, self.target_critic = nets
        self.target_map = target_map
        self._print_num_params(self.actor.parameters(), name="actor")
        self._print_num_params(self.critic.parameters(), name="critic")

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()  # TODO

        self._print_num_params(self.actor.parameters(), name="actor")

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True,
            weight_decay=0.1
        )
        self.optimizer_critic = optim.AdamW(
            self.critic.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer_actor, self.optimizer_critic]
        self.schedulers = self._prepare_schedulers()
        self.criterion_critic = nn.SmoothL1Loss()
        self.type = type

    def _training_step(self, batch, step_i, gamma):
        state, action, reward, next_state, candidates, not_done = batch
        batch_size = state.shape[0]

        q_value = self.critic(state, action)
        q_value = q_value.squeeze(-1)

        with torch.no_grad():
            if self.type == "default":
                proto_next_action = self.target_actor(next_state)
            elif self.type == "lstm":
                c_lstm = candidates.clone().detach()
                c_lstm = torch.nan_to_num(c_lstm, nan=0.0, neginf=0.0)
                proto_next_action = self.target_actor(next_state, c_lstm)

            best_candidates = self._find_closest_candidates(
                candidates,
                proto_next_action,
                batch_size
            )
            k_values = self._get_k_values(best_candidates)
            batch_next_q_value = torch.empty(batch_size, device=self.device)
            for j in range(batch_size):
                if torch.any(torch.isnan(best_candidates[j][0])) or \
                   torch.any(torch.isinf(best_candidates[j][0])):
                    batch_next_q_value[j] = 0
                    continue

                k = k_values[j]
                next_state_repeated = next_state[j].repeat(k, 1)

                next_q_value = self.target_critic(
                    next_state_repeated,
                    best_candidates[j][:k]
                )
                next_q_value = next_q_value.squeeze(-1)
                batch_next_q_value[j] = torch.max(next_q_value)

            q_target = reward + (gamma * batch_next_q_value * not_done)

        loss_critic = self.criterion_critic(q_value, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        if self.type == "default":
            proto_action = self.actor(state)
        elif self.type == "lstm":
            prev_candidates = self._get_prev_candidates(
                c_lstm,
                action,
                batch_size
            )
            proto_action = self.actor(
                state,
                prev_candidates
            )

        action_sim_loss = torch.cdist(action, proto_action) * q_value.detach()
        action_sim_loss = action_sim_loss.mean()

        action_sum_loss = (proto_action.square().sum(dim=1).mean() - 6.769)**2
        loss_actor = action_sim_loss + action_sum_loss
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

    def _find_closest_candidates(self, candidates, proto_action, batch_size):
        proto_action = proto_action.unsqueeze(1)
        distances = torch.cdist(candidates, proto_action)
        distances = distances.squeeze(-1)
        sort_order = torch.argsort(distances, dim=1, descending=False)
        best_candidates = candidates[
            torch.arange(batch_size).reshape(-1, 1),
            sort_order,
        ]
        return best_candidates

    def _get_k_values(self, best_candidates):
        k_values = [
            (c != torch.full((1, 768), -torch.inf,
             device=self.device)).all(1).sum().item()
            for c in best_candidates
        ]
        k_values = [
            k if round(k * 0.25) == 0
            else round(k * 0.25)
            for k in k_values
        ]
        return k_values

    def _get_prev_candidates(self, candidates, action, batch_size):
        prev_candidates = torch.empty(
            candidates.shape[0],
            candidates.shape[1]+1,
            candidates.shape[2],
            device=self.device
        )
        for i in range(batch_size):
            prev_candidates[i] = torch.vstack([action[i], candidates[i]])
            prev_candidates[i] = prev_candidates[i][
                torch.randperm(prev_candidates[i].size()[0])
            ]
