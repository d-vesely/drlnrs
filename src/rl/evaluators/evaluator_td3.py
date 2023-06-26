import torch

from ..algorithms.td3 import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorTD3(_EvaluatorBase):
    def __init__(self, development, model_name, device, test_model="final"):
        super().__init__(
            development,
            model_name,
            device,
            test_model
        )

    def set_evaluatee(self, type):
        nets = get_evaluatee(
            self.config_model,
            self.device,
            type
        )
        self.actor, = nets
        self.type = type

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(torch.load(checkpoint))
        self.actor.eval()

    def _get_desc_sort_order(self, state, candidates):
        if self.type == "default":
            action, _ = self.actor(state)
        elif self.type == "lstm":
            action, _ = self.actor(state, candidates)

        candidate_scores = (action * candidates)
        candidate_scores = candidate_scores.sum(dim=-1)

        # Argsort q-values in descending order
        desc_sort_order = torch.argsort(candidate_scores, descending=True)
        return desc_sort_order
