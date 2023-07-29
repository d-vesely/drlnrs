import torch
import torch.nn as nn
import torch.optim as optim

from ..algorithms.ddpg import get_trainee
from .trainer_base import _TrainerBase


class TrainerDDPG(_TrainerBase):
    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params,
                 ep_rm_path=None, seed=None):
        super().__init__(
            model_name,
            device,
            pos_rm_path,
            neg_rm_path,
            encoder_params,
            learning_params,
            model_params,
            ep_rm_path,
            seed
        )

    def set_trainee(self):
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
        )
        self.actor, self.target_actor, \
            self.critic, self.target_critic = nets
        self.target_map = target_map
        self._print_num_params(self.actor.parameters(), name="actor")
        self._print_num_params(self.critic.parameters(), name="critic")

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

    def _training_step(self, batch, step_i, gamma, print_q):
        state, action, reward, next_state, candidates, not_done = batch
        batch_size = state.shape[0]

        q_value = self.critic(state, action)
        q_value = q_value.squeeze(-1)

        with torch.no_grad():
            proto_next_action = self.target_actor(next_state)
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

        proto_action = self.actor(state)

        action_sim_loss = torch.cdist(
            action, proto_action)  # * q_value.detach()
        action_sim_loss = action_sim_loss.mean()

        action_sum_loss = (proto_action.square().sum(dim=1).mean() - 6.769)**2
        loss_actor = action_sim_loss  # + action_sum_loss
        loss_actor = -self.critic(state, action).mean()
        print(proto_action)
        print(loss_actor)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        if print_q:
            print("[INFO] example Q values: ")
            print(q_value)

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
