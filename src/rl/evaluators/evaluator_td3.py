import torch

from ..algorithms.td3 import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorTD3(_EvaluatorBase):
    """Evaluator for TD3"""

    def __init__(self, development, model_name, device, seed=None, test_ckpt=None):
        self.ac = True
        super().__init__(
            development,
            model_name,
            device,
            seed,
            test_ckpt
        )

    def set_evaluatee(self, involved):
        """Prepare evaluatee"""
        nets = get_evaluatee(
            self.config_model,
            self.device,
            involved
        )
        self.involved = involved
        # Set descending sort order depending on whether the actor, the critic
        # or both are used for evaluation
        if involved == "a":
            self.actor, = nets
            self._get_desc_sort_order = self._get_desc_sort_order_a
        elif involved == "c":
            self.critic, = nets
            self._get_desc_sort_order = self._get_desc_sort_order_c
        elif involved == "ac":
            self.actor, self.critic = nets
            self._get_desc_sort_order = self._get_desc_sort_order_ac

    def _load_checkpoint(self, checkpoint):
        """Load model checkpoint into nets"""
        actor_ckpt, critic_ckpt = checkpoint
        # Load checkpoints depending on whether the actor, the critic
        # or both are used for evaluation
        if self.involved == "a":
            self.actor.load_state_dict(torch.load(actor_ckpt))
            self.actor.eval()
        elif self.involved == "c":
            self.critic.load_state_dict(torch.load(critic_ckpt))
            self.critic.eval()
        elif self.involved == "ac":
            self.actor.load_state_dict(torch.load(actor_ckpt))
            self.actor.eval()
            self.critic.load_state_dict(torch.load(critic_ckpt))
            self.critic.eval()

    def _get_desc_sort_order_a(self, state, candidates):
        """Get recommendation order from best to worst candidate (actor)"""
        # Compute proto-action
        proto_action = self.actor(state)
        proto_action = proto_action.unsqueeze(0)
        # Compute distances to candidates
        distances = torch.cdist(candidates, proto_action)
        distances = distances.reshape(-1)
        # Create recommendation order from distances
        desc_sort_order = torch.argsort(distances, descending=False)
        return desc_sort_order

    def _get_desc_sort_order_c(self, state, candidates):
        """Get recommendation order from best to worst candidate (critic)"""
        # Repeat state for each candidate
        state_repeated = state.repeat(len(candidates), 1)
        # Get q-values and create sort order for candidates
        q_values = self.critic(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order

    def _get_desc_sort_order_ac(self, state, candidates):
        """Get recommendation order from best to worst candidate (actor+critic)"""
        # Compute proto-action
        proto_action = self.actor(state)
        proto_action = proto_action.unsqueeze(0)
        # Compute distances to candidates
        distances = torch.cdist(candidates, proto_action)
        distances = distances.reshape(-1)
        # Create recommendation order from distances
        desc_sort_order_actor = torch.argsort(distances, descending=False)

        # Half of closest candidates are evaluated by critic
        k = round(0.5 * len(candidates))
        if k == 0:
            k = len(candidates)
        top_k_sort_order = desc_sort_order_actor[:k]
        critic_candidates = candidates[top_k_sort_order]

        # Repeat state for each candidate
        state_repeated = state.repeat(len(critic_candidates), 1)

        # Get q-values and create sort order for candidates
        q_values = self.critic(state_repeated, critic_candidates)
        q_values = q_values.reshape(-1)

        # Join two recommendation orders
        desc_sort_order_critic = torch.argsort(q_values, descending=True)
        desc_sort_order = torch.concatenate((
            top_k_sort_order[desc_sort_order_critic],
            desc_sort_order_actor[k:]
        ))
        return desc_sort_order

    def _get_desc_sort_order_indirect(self, state, candidates):
        """Get recommendation order for indirect approach"""
        # Get scoring vector
        action, _ = self.actor(state)

        # Compute descending sort order from candidate scores
        candidate_scores = (action * candidates)
        candidate_scores = candidate_scores.sum(dim=-1)
        desc_sort_order = torch.argsort(candidate_scores, descending=True)
        return desc_sort_order
