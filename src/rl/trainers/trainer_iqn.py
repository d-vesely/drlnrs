import torch
import torch.optim as optim

from ..algorithms.iqn import get_trainee, get_next_action
from ..algorithms.dist_rl_utils import get_huber_loss
from .trainer_base import _TrainerBase


class TrainerIQN(_TrainerBase):
    """Trainer for IQN"""

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
        """Prepare trainee"""
        # Get model architecture and online-target map
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
        )
        self.dqn, self.target_dqn = nets
        self.target_map = target_map

        # Print number of trainable parameters
        self._print_num_params(self.dqn.parameters(), name="DQN")

        # Prepare optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()

        # Load hyperparameter number of quantiles
        self.n_quantiles = self.config_model["net_params"]["n_quantiles"]

    def _training_step(self, batch, step_i, gamma, print_q):
        """Single training step"""
        # Load batch
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]
        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        # Compute quantiles, keep sampled taus
        quantiles, tau = self.dqn(state, item)
        quantiles = quantiles.squeeze(1)
        quantiles = quantiles.unsqueeze(-1)

        with torch.no_grad():
            # Copy next state for each candidate
            next_state_rep = next_state.unsqueeze(1).repeat(
                1, n_candidates, 1
            )
            # Compute next state quantiles
            next_quantiles, _ = self.target_dqn(next_state_rep, candidates)
            # Get next quantiles for best action
            _, next_quantiles = get_next_action(
                next_quantiles,
                self.n_quantiles
            )
            next_quantiles = next_quantiles.squeeze(1)

            if torch.any(torch.isnan(next_quantiles)):
                next_quantiles = torch.nan_to_num(next_quantiles, nan=0.0)

            # Apply reward and discount to get next quantiles
            next_quantiles = reward + (gamma * next_quantiles * not_done)
            next_quantiles = next_quantiles.unsqueeze(1)

        # Get temporal difference error and compute huber loss
        td_error = next_quantiles - quantiles
        huber_loss = get_huber_loss(td_error)
        # Use sampled tau to compute loss
        loss = huber_loss * torch.abs(tau - (td_error < 0).float())
        loss = loss.sum(dim=1).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
