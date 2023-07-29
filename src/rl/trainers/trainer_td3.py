import torch
import torch.nn as nn
import torch.optim as optim

from ..algorithms.td3 import get_trainee
from .trainer_base import _TrainerBase


class TrainerTD3(_TrainerBase):
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
            self.critic_1, self.target_critic_1, \
            self.critic_2, self.target_critic_2 = nets
        self.target_map = target_map
        self._print_num_params(self.actor.parameters(), name="actor")
        self._print_num_params(self.critic_1.parameters(), name="critic (x2)")

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True,
            weight_decay=0.1
        )
        self.optimizer_critic = optim.AdamW(
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer_actor, self.optimizer_critic]
        self.schedulers = self._prepare_schedulers()
        self.criterion_critic = nn.SmoothL1Loss()

    def _training_step(self, batch, step_i, gamma, print_q):
        state, item, reward, next_state, candidates, not_done = batch

        action = (item / torch.linalg.norm(item)) * reward[0]
        print((action**2).sum())
        _, state_embedding = self.actor(state)

        q_value_1 = self.critic_1(state_embedding.detach(), action)
        q_value_2 = self.critic_2(state_embedding.detach(), action)
        q_value_1 = q_value_1.squeeze(-1)
        q_value_2 = q_value_2.squeeze(-1)
        if print_q:
            print("[INFO] example Q values: ")
            print(q_value_1)

        with torch.no_grad():
            next_action, next_state_embedding = self.target_actor(
                next_state
            )

            next_q_value_1 = self.target_critic_1(
                next_state_embedding,
                next_action
            )
            next_q_value_2 = self.target_critic_2(
                next_state_embedding,
                next_action
            )
            next_q_value_1 = next_q_value_1.squeeze(-1)
            next_q_value_2 = next_q_value_2.squeeze(-1)
            next_q_value = torch.min(next_q_value_1, next_q_value_2)
            target = reward + (gamma * next_q_value * not_done)

        loss_critic_1 = self.criterion_critic(q_value_1, target)
        loss_critic_2 = self.criterion_critic(q_value_2, target)
        loss_critic = loss_critic_1 + loss_critic_2

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        if step_i % 2 == 0:
            gen_action, se = self.actor(state)

            # TODO try different losses
            # TODO randomly pick q_value1/2
            dist_loss = (torch.cdist(action, gen_action) * q_value_1.detach())
            dist_loss = dist_loss.mean()

            # item_score = (item * gen_action).sum(dim=-1)
            # if reward[0] == 1:
            #     item_score = (-1 * item_score)
            # item_score_loss = (item_score * q_value.detach()).mean()

            sum_loss = (((gen_action ** 2).sum(dim=1) - 1)**2)
            sum_loss = sum_loss.mean()

            loss_actor = dist_loss + sum_loss
            loss_actor = -self.critic_1(se, gen_action).mean() + sum_loss
            print(gen_action)
            print(loss_actor)

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
