import torch
import torch.nn as nn
import torch.optim as optim

from ..algorithms.sac import get_trainee, get_action_probs
from trainer_base import _TrainerBase


class TrainerSAC(_TrainerBase):
    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params, rm_episodic_path=""):
        super(_TrainerBase).__init__(
            model_name,
            device,
            pos_rm_path,
            neg_rm_path,
            encoder_params,
            learning_params,
            model_params
        )

    def set_trainee(self):
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
        )
        self.actor, \
            self.soft_critic_1, self.target_soft_critic_1, \
            self.soft_critic_2, self.target_soft_critic_2 = nets
        self.target_map = target_map

        self._print_num_params(self.actor.parameters(), name="actor")
        self._print_num_params(
            self.soft_critic_1.parameters(),
            name="critic (x2)"
        )

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizer_critic = optim.AdamW(
            list(self.soft_critic_1.parameters()) +
            list(self.soft_critic_2.parameters()),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer_actor, self.optimizer_critic]
        self.schedulers = self._prepare_schedulers()
        self.criterion_critic = nn.SmoothL1Loss()

    def _training_step(self, batch, step_i, gamma):
        state, item, reward, next_state, candidates, not_done = batch
        batch_size = state.shape[0]
        n_candidates = candidates.shape[1]

        # Compute current q-value
        q_value_1 = self.softcritic_1(state, item)
        q_value_2 = self.softcritic_2(state, item)

        # Get q-values for taken actions # TODO
        acts = torch.tensor([0 if r == -1 else 1 for r in reward])

        q_value_1 = q_value_1[torch.arange(batch_size), reward]
        q_value_2 = q_value_2[torch.arange(batch_size), reward]

        with torch.no_grad():
            # Copy next state for each candidate
            next_state_rep = next_state.unsqueeze(1).repeat(
                1, n_candidates, 1
            )

            # Compute q-values for all candidates
            next_q_values_1 = self.target_softcritic_1(
                next_state_rep,
                candidates
            )
            next_q_values_2 = self.target_softcritic_2(
                next_state_rep,
                candidates
            )

            # Set q-values produced by padded-candidates to large negative number
            next_q_values_1 = torch.nan_to_num(next_q_values_1, nan=-10000)
            next_q_values_2 = torch.nan_to_num(next_q_values_2, nan=-10000)

            # TODO randomly pick net
            # Find best candidate indices
            best_candidates_idx = torch.argmax(next_q_values_1[:, :, 1], dim=1)
            # Find max q-value and compute target
            max_next_q_value_1 = next_q_values_1[
                torch.arange(batch_size),
                best_candidates_idx
            ]
            max_next_q_value_2 = next_q_values_2[
                torch.arange(batch_size),
                best_candidates_idx
            ]

            # Compute action logits for best candidates
            best_candidates = candidates[
                torch.arange(batch_size),
                best_candidates_idx
            ]
            action_logits = self.actor(next_state, best_candidates)
            # Some logits can still be NaN, if the candidates are empty (i.e. done)
            # These values will not affect the target, because not_done == 0
            if torch.any(torch.isnan(action_logits)):
                action_logits = torch.nan_to_num(action_logits, nan=0.0)

            # Get action probabilities and log-probabilities
            action_probs, log_probs = get_action_probs(action_logits)

            min_next_q_value = torch.min(
                max_next_q_value_1,
                max_next_q_value_2
            )

            # Compute next q-value and target
            next_q_value = action_probs * \
                (min_next_q_value - 0.2 * log_probs)
            q_target = reward + (gamma * next_q_value * not_done)

        # Update critics together
        loss_1 = self.criterion_critic(q_value_1, q_target)
        loss_2 = self.criterion_critic(q_value_2, q_target)
        loss = loss_1 + loss_2
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        # Get action probabilities and log-probabilities for current state and item
        action_logits = self.actor(state, item)
        action_probs, log_probs = get_action_probs(action_logits)

        with torch.no_grad():  # TODO do not recompute?
            q_value_1 = self.softcritic_1(state, item)
            q_value_2 = self.softcritic_2(state, item)
            min_q_value = torch.min(q_value_1, q_value_2)

        # Update actor
        loss_actor = action_probs * \
            (0.2 * log_probs - min_q_value)
        loss_actor = loss_actor.mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
