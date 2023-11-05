import torch
import torch.optim as optim

from ..algorithms.fqf import get_trainee, get_next_action
from ..algorithms.dist_rl_utils import get_huber_loss
from .trainer_base import _TrainerBase


class TrainerFQF(_TrainerBase):
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
        self.dqn, self.target_dqn, self.fpn = nets
        self.target_map = target_map

        # Print number of trainable parameters
        self._print_num_params(self.dqn.parameters(), name="DQN")
        self._print_num_params(self.fpn.parameters(), name="FPN")

        # Prepare optimizers and schedulers
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizer_fpn = optim.RMSprop(
            self.fpn.parameters(),
            lr=1e-9,
            alpha=0.95,
            eps=1e-5
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()

    def _training_step(self, batch, step_i, gamma, print_q):
        """Single training step"""
        # Load batch
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]
        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        # Embed state + item and send through FPN
        embedding = self.dqn.get_embedding(state, item)
        # Get tau and tau-hat from fraction proposal network
        tau, tau_hat, entropy = self.fpn(embedding.detach())
        quantiles = self.dqn.get_quantiles(embedding, tau_hat)
        quantiles = quantiles.squeeze(1)

        # Compute loss for FPN
        with torch.no_grad():
            FZ_tau = self.dqn.get_quantiles(
                embedding.detach(), tau[:, 1:-1]
            )
            FZ_tau = FZ_tau.squeeze(1)

        fraction_loss = self.fpn.get_loss(quantiles.detach(), FZ_tau, tau)
        # fraction_loss += (1e-3 * entropy.mean())

        with torch.no_grad():
            # Copy next state for each candidate
            next_state_rep = next_state.unsqueeze(1).repeat(1, n_candidates, 1)
            # Embed next states + items
            next_embedding = self.target_dqn.get_embedding(
                next_state_rep,
                candidates
            )
            # Get tau and tau-hat from fraction proposal network
            next_tau, next_tau_hat, entropy = self.fpn(next_embedding)
            next_quantiles = self.target_dqn.get_quantiles(
                next_embedding,
                next_tau_hat
            )
            # Get next quantiles for best action
            _, next_quantiles = get_next_action(next_quantiles, next_tau)
            next_quantiles = next_quantiles.squeeze(1)

            if torch.any(torch.isnan(next_quantiles)):
                next_quantiles = torch.nan_to_num(next_quantiles, nan=0.0)

            # Apply reward and discount to get next quantiles
            next_quantiles = reward + (gamma * next_quantiles * not_done)
            next_quantiles = next_quantiles.unsqueeze(1)

        # Get temporal difference error and compute huber loss
        quantiles = quantiles.unsqueeze(-1)
        td_error = next_quantiles - quantiles
        huber_loss = get_huber_loss(td_error)
        loss = huber_loss * \
            torch.abs(tau_hat.unsqueeze(-1) - (td_error < 0).float())
        loss = loss.sum(dim=1).mean(dim=1).mean()

        self.optimizer_fpn.zero_grad()
        fraction_loss.backward(retain_graph=True)
        self.optimizer_fpn.step()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
