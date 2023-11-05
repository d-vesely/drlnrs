import torch
import torch.optim as optim

from ..algorithms.c51 import get_evaluatee, get_q_values
from .evaluator_base import _EvaluatorBase


class EvaluatorC51(_EvaluatorBase):
    """Evaluator for C51"""

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
        """Prepare evaluatee"""
        nets = get_evaluatee(
            self.config_model,
            self.device
        )
        self.dqn, = nets

    def _load_checkpoint(self, checkpoint):
        """Load model checkpoint into net"""
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

    def _get_desc_sort_order(self, state, candidates):
        """Get recommendation order from best to worst candidate"""
        # Repeat state for each candidate
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)

        # Get pmfs
        pmfs = self.dqn(state_repeated, candidates)

        # Get q-values for given pmfs
        q_values = get_q_values(pmfs, self.dqn.supports)
        q_values = q_values.squeeze(-1)

        # Create sort order for candidates
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
