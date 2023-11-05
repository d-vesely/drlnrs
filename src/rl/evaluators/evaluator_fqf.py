import torch

from ..algorithms.fqf import get_evaluatee, get_q_values_eval
from .evaluator_base import _EvaluatorBase


class EvaluatorFQF(_EvaluatorBase):
    """Evaluator for FQF"""

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
        # Model checkpoints include FPN checkpoints
        self.model_ckpts = self.model_ckpts[0:len(self.model_ckpts)//2]
        nets = get_evaluatee(
            self.config_model,
            self.device,
        )
        self.dqn, self.fpn = nets
        # Number of quantiles is a hyperparameter
        self.n_quantiles = self.config_model["net_params"]["n_quantiles"]

    def _load_checkpoint(self, checkpoint):
        """Load model checkpoint into nets"""
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

        self.fpn.load_state_dict(torch.load(
            checkpoint.replace("dqn", "fpn")))
        self.fpn.eval()

    def _get_desc_sort_order(self, state, candidates):
        """Get recommendation order from best to worst candidate"""
        # Repeat state for each candidate
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)

        # Get state embedding
        embedding = self.dqn.get_embedding(state_repeated, candidates)

        # Get fraction proposals
        tau, tau_hat, entropy = self.fpn(embedding.detach())

        # Get quantiles
        quantiles = self.dqn.get_quantiles(embedding, tau_hat)

        # Get q-values for quantiles
        q_values = get_q_values_eval(quantiles, tau)
        q_values = q_values.squeeze(-1)

        # Create sort order for candidates
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
