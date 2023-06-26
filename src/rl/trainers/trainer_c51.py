import torch
import torch.optim as optim

from ..algorithms.c51 import get_trainee, get_next_action
from .trainer_base import _TrainerBase


class TrainerC51(_TrainerBase):
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

        self._print_num_params(self.dqn.parameters(), name="DQN_C51")

        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()

        self.n_atoms = self.config_model["net_params"]["n_atoms"]
        self.v = self.config_model["net_params"]["v"]

    def _training_step(self, batch, step_i, gamma, print_q):
        state, item, reward, next_state, candidates, not_done = batch
        n_candidates = candidates.shape[1]

        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        pmfs = self.dqn(state, item)
        pmfs = pmfs.squeeze(1)
        # _, pmfs = act(pmfs, self.dqn.supports)
        # pmfs = pmfs.clamp(min=1e-5, max=(1 - 1e-5))

        with torch.no_grad():
            # Copy next state for each candidate
            next_state_rep = next_state.unsqueeze(1).repeat(1, n_candidates, 1)
            next_pmfs = self.target_dqn(next_state_rep, candidates)
            _, next_pmfs = get_next_action(next_pmfs, self.target_dqn.supports)
            if torch.any(torch.isnan(next_pmfs)):
                next_pmfs = torch.nan_to_num(next_pmfs, nan=0.0)

            next_supports = reward + \
                (gamma * self.target_dqn.supports * not_done)

            delta_z = self.target_dqn.supports[1] - \
                self.target_dqn.supports[0]

            tz = next_supports.clamp(-self.v, self.v)
            b = (tz - (-self.v)) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            dml = (u + (l == u).float() - b) * next_pmfs
            dmu = (b - l) * next_pmfs
            target_pmfs = torch.zeros(next_pmfs.shape, device=self.device)
            for j in range(target_pmfs.size(0)):
                target_pmfs[j].index_add_(0, l[j].long(), dml[j])
                target_pmfs[j].index_add_(0, u[j].long(), dmu[j])

        loss = -(target_pmfs * pmfs.clamp(min=1e-6,
                                          max=1-1e-6).log())
        loss = loss.sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
