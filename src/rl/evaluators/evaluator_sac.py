import torch

from ..algorithms.sac import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorSAC(_EvaluatorBase):
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

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(torch.load(checkpoint))
        self.actor.eval()

    def _get_desc_sort_order(self, state, candidates):
        state_repeated = state.repeat(len(candidates), 1)
        action_logits = self.actor(state_repeated, candidates)
        desc_sort_order = torch.argsort(action_logits[:, 1], descending=True)
        return desc_sort_order
