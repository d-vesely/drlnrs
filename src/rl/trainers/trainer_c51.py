import torch
import torch.optim as optim

from ..algorithms.c51 import get_trainee, get_next_action
from .trainer_base import _TrainerBase


class TrainerC51(_TrainerBase):
    """Trainer for C51"""

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
            self.device
        )
        self.dqn, self.target_dqn = nets
        self.target_map = target_map

        # Print number of trainable parameters
        self._print_num_params(self.dqn.parameters(), name="DQN_C51")

        # Prepare optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()

        # Load hyperparameters n_atoms and v (vmin = vmax = v)
        self.n_atoms = self.config_model["net_params"]["n_atoms"]
        self.v = self.config_model["net_params"]["v"]

    def _training_step(self, batch, step_i, gamma, print_q):
        """Single training step"""
        # Load batch
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]
        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        # Compute PMFs
        pmfs = self.dqn(state, item)
        pmfs = pmfs.squeeze(1)

        with torch.no_grad():
            # Copy next state for each candidate
            next_state_rep = next_state.unsqueeze(1).repeat(
                1, n_candidates, 1
            )
            # Compute next state PMFs
            next_pmfs = self.target_dqn(next_state_rep, candidates)
            # Get next pmf for best action
            _, next_pmfs = get_next_action(next_pmfs, self.target_dqn.supports)
            if torch.any(torch.isnan(next_pmfs)):
                next_pmfs = torch.nan_to_num(next_pmfs, nan=0.0)

            # Apply reward and discount factor to get new supports
            next_supports = reward + \
                (gamma * self.target_dqn.supports * not_done)

            # Get stepsize between supports
            delta_z = self.target_dqn.supports[1] - \
                self.target_dqn.supports[0]

            # Projection step
            Tz = next_supports.clamp(-self.v, self.v)
            # Project positions into [0, n_atoms-1]
            b = (Tz - (-self.v)) / delta_z
            # Get lower and upper position
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            # Distribute probabilities proportionally
            dml = (u + (l == u).float() - b) * next_pmfs
            dmu = (b - l) * next_pmfs
            target_pmfs = torch.zeros(next_pmfs.shape, device=self.device)
            # Sum over all probability distributions
            for j in range(target_pmfs.size(0)):
                target_pmfs[j].index_add_(0, l[j].long(), dml[j])
                target_pmfs[j].index_add_(0, u[j].long(), dmu[j])

        # Cross-entropy loss
        loss = -(target_pmfs * pmfs.clamp(min=1e-6,
                                          max=1-1e-6).log())
        loss = loss.sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
