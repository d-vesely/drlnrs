import torch
import torch.optim as optim

from ..algorithms.fqf import get_trainee, act, get_next_action
from ..algorithms.dist_rl_utils import get_huber_loss
from .trainer_base import _TrainerBase


class TrainerFQF(_TrainerBase):
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
        self.dqn, self.target_dqn, self.fpn = nets
        self.target_map = target_map

        self._print_num_params(self.dqn.parameters(), name="DQN")
        self._print_num_params(self.fpn.parameters(), name="FPN")

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
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]

        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        embedding = self.dqn.get_embedding(state, item)
        tau, tau_hat, entropy = self.fpn(embedding.detach())
        F_Z = self.dqn.get_quantiles(embedding, tau_hat)
        # acts = torch.tensor([0 if r == -1 else 1 for r in reward])
        # _, quantiles = act(F_Z, action=reward)
        quantiles = F_Z.squeeze(1)

        with torch.no_grad():
            Z_tau = self.dqn.get_quantiles(
                embedding.detach(), tau[:, 1:-1])
            # _, FZ_tau = act(Z_tau, action=reward)
            FZ_tau = Z_tau.squeeze(1)

        fraction_loss = self.fpn.get_loss(quantiles.detach(), FZ_tau, tau)
        # fraction_loss += (1e-3 * entropy.mean())

        with torch.no_grad():
            next_state_rep = next_state.unsqueeze(1).repeat(1, n_candidates, 1)
            next_embedding = self.target_dqn.get_embedding(
                next_state_rep,
                candidates
            )
            next_tau, next_tau_hat, entropy = self.fpn(next_embedding)

            F_Z_next = self.target_dqn.get_quantiles(
                next_embedding,
                next_tau_hat
            )
            _, next_quantiles = get_next_action(F_Z_next, next_tau)
            next_quantiles = next_quantiles.squeeze(1)

            if torch.any(torch.isnan(next_quantiles)):
                next_quantiles = torch.nan_to_num(next_quantiles, nan=0.0)

            next_quantiles = reward + (gamma * next_quantiles * not_done)
            next_quantiles = next_quantiles.unsqueeze(1)

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
