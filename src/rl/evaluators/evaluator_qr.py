import torch

from ..algorithms.qr import get_evaluatee, get_q_values
from .evaluator_base import _EvaluatorBase


class EvaluatorQR(_EvaluatorBase):
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
        nets = get_evaluatee(
            self.config_model,
            self.device
        )
        self.dqn, = nets
        self.n_quantiles = self.config_model["net_params"]["n_quantiles"]

    def _load_checkpoint(self, checkpoint):
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

    def _get_desc_sort_order(self, state, candidates):
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        quantiles = self.dqn(state_repeated, candidates)
        q_values = get_q_values(quantiles, self.n_quantiles)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
