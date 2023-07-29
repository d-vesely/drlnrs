import torch

from ..algorithms.td3 import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorTD3(_EvaluatorBase):
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
            self.device,
        )
        self.actor, = nets
        self._get_desc_sort_order = self._get_desc_sort_order

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(torch.load(checkpoint))
        self.actor.eval()

    def _get_desc_sort_order(self, state, candidates):
        action, _ = self.actor(state)

        candidate_scores = (action * candidates)
        candidate_scores = candidate_scores.sum(dim=-1)

        # Argsort q-values in descending order
        desc_sort_order = torch.argsort(candidate_scores, descending=True)
        return desc_sort_order
