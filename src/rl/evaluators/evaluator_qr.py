import torch

from ..algorithms.qr import get_evaluatee, get_q_values
from .evaluator_base import _EvaluatorBase


class EvaluatorQR(_EvaluatorBase):
    """Evaluator for QR-DQN"""

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
        # Number of quantiles is a hyperparameter
        self.n_quantiles = self.config_model["net_params"]["n_quantiles"]

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

        # Get quantiles
        quantiles = self.dqn(state_repeated, candidates)

        # Get q-values for quantiles
        q_values = get_q_values(quantiles, self.n_quantiles)

        # Create sort order for candidates
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
