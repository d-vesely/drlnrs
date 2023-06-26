import torch

from ..algorithms.fqf import get_evaluatee, get_q_values_eval
from .evaluator_base import _EvaluatorBase


class EvaluatorFQF(_EvaluatorBase):
    def __init__(self, development, model_name, device, seed=None,
                 test_ckpt=None):
        super().__init__(
            development,
            model_name,
            device,
            seed,
            test_ckpt
        )

    def set_evaluatee(self):
        self.model_ckpts = self.model_ckpts[0:len(self.model_ckpts)//2]
        nets = get_evaluatee(
            self.config_model,
            self.device,
        )
        self.dqn, self.fpn = nets
        self.n_quantiles = self.config_model["net_params"]["n_quantiles"]

    def _load_checkpoint(self, checkpoint):
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

        self.fpn.load_state_dict(torch.load(
            checkpoint.replace("dqn", "fpn")))  # TODO
        self.fpn.eval()

    def _get_desc_sort_order(self, state, candidates):
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        embedding = self.dqn.get_embedding(state_repeated, candidates)
        tau, tau_hat, entropy = self.fpn(embedding.detach())  # TODO entropy
        Z = self.dqn.get_quantiles(embedding, tau_hat)
        q_values = get_q_values_eval(Z, tau)
        q_values = q_values.squeeze(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
