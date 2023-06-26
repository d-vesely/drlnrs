import torch

from ..algorithms.dqn import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorDQN(_EvaluatorBase):
    def __init__(self, development, model_name, device, seed=None, test_ckpt=None):
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
        self._get_desc_sort_order = self._get_desc_sort_order

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
        q_values = self.dqn(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
