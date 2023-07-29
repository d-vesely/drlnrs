import torch

from ..algorithms.reinforce import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorREINFORCE(_EvaluatorBase):
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
        self.actor, = nets
        self._get_desc_sort_order = self._get_desc_sort_order

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(torch.load(checkpoint))
        self.actor.eval()

    def _get_desc_sort_order(self, state, candidates):
        state_repeated = state.repeat(len(candidates), 1)
        action_probs = self.actor(state_repeated, candidates)
        desc_sort_order = torch.argsort(action_probs[:, 1], descending=True)
        return desc_sort_order
