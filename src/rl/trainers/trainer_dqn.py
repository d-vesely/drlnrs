import torch
import torch.nn as nn
import torch.optim as optim

from ..algorithms.dqn import get_trainee
from .trainer_base import _TrainerBase


class TrainerDQN(_TrainerBase):
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
            self.device
        )
        self.dqn, self.target_dqn = nets
        self.target_map = target_map
        self._print_num_params(self.dqn.parameters(), name="DQN")

        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()
        self.criterion = nn.SmoothL1Loss()
        double_learning = self.config_model["double_learning"]
        if double_learning:
            self._training_step = self._training_step_double
        elif self.config_model["type"] == "dueling":
            self._training_step = self._training_step_duel
        else:
            self._training_step = self._training_step

    def _training_step(self, batch, step_i, gamma, print_q):
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]

        # Compute current q-value
        q_value = self.dqn(state, item)
        q_value = q_value.squeeze(-1)

        with torch.no_grad():
            # Copy next state for each candidate
            rep_shape = self._get_rep_shape(state.shape, n_candidates)
            next_state_rep = next_state.unsqueeze(1).repeat(*rep_shape)

            # Compute q-values for all candidates
            next_q_value = self.target_dqn(next_state_rep, candidates)
            next_q_value = next_q_value.squeeze(-1)

            # TODO
            # Set q-values produced by padded-candidates to large negative number
            next_q_value = torch.nan_to_num(next_q_value, nan=-10000)
            # Find max q-value and compute target
            max_next_q_value = torch.max(next_q_value, dim=1).values
            q_target = reward + (gamma * max_next_q_value * not_done)

        # Update DQN
        if print_q:
            print("[INFO] example Q values: ")
            print(q_value)
        loss = self.criterion(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _training_step_duel(self, batch, step_i, gamma, print_q):
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]

        with torch.no_grad():
            # Copy next state for each candidate
            rep_shape = self._get_rep_shape(state.shape, n_candidates)
            next_state_rep = next_state.unsqueeze(1).repeat(*rep_shape)

            # Compute q-values for all candidates
            next_vals, next_adv = self.target_dqn(next_state_rep, candidates)
            next_q_value = (next_adv - next_adv.nanmean(dim=1).unsqueeze(1)) + \
                next_vals.nanmean(dim=1).unsqueeze(1)
            next_q_value = next_q_value.squeeze(-1)

            # Set q-values produced by padded-candidates to large negative number
            next_q_value = torch.nan_to_num(next_q_value, nan=-10000)
            # Find max q-value and compute target
            max_next_q_value = torch.max(next_q_value, dim=1).values
            q_target = reward + (gamma * max_next_q_value * not_done)

        # Compute current q-value
        value, adv = self.dqn(state, item)
        q_value = value + \
            (adv - next_adv.nanmean(dim=1).nan_to_num(nan=0))
        q_value = q_value.squeeze(-1)

        # Update DQN
        if print_q:
            print("[INFO] example Q values: ")
            print(q_value)
        loss = self.criterion(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _training_step_double(self, batch, step_i, gamma, print_q):
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]

        # Compute current q-value
        q_value = self.dqn(state, item)
        q_value = q_value.squeeze(-1)

        with torch.no_grad():
            # Copy next state for each candidate
            rep_shape = self._get_rep_shape(state.shape, n_candidates)
            next_state_rep = next_state.unsqueeze(1).repeat(*rep_shape)

            # Compute q-values for all candidates
            next_q_value = self.dqn(next_state_rep, candidates)
            next_q_value = next_q_value.squeeze(-1)

            # TODO
            # Set q-values produced by padded-candidates to large negative number
            next_q_value = torch.nan_to_num(next_q_value, nan=-10000)
            if torch.any(torch.isnan(next_q_value)):
                print(next_q_value)
                return
            # Find max q-value and compute target
            argmax_next_q_value = torch.argmax(next_q_value, dim=1)
            best_candidates = candidates[
                torch.arange(candidates.shape[0]),
                argmax_next_q_value
            ]
            max_next_q_value = self.target_dqn(next_state, best_candidates)
            max_next_q_value = max_next_q_value.squeeze(-1)
            max_next_q_value[(not_done == 0)] = 0.0
            q_target = reward + (gamma * max_next_q_value * not_done)

        # Update DQN
        if print_q:
            print("[INFO] example Q values: ")
            print(q_value)
        loss = self.criterion(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_rep_shape(self, state_shape, n_candidates):
        rep_shape = [1, n_candidates, 1]
        if len(state_shape) == 3:
            rep_shape.append(1)
        return rep_shape
