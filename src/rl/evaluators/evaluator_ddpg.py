import torch

from ..algorithms.ddpg import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorDDPG(_EvaluatorBase):
    def __init__(self, development, model_name, device, test_model="final"):
        super().__init__(
            development,
            model_name,
            device,
            test_model
        )

    def set_evaluatee(self, type, involved):
        nets = get_evaluatee(
            self.config_model,
            self.device,
            type,
            involved
        )
        if involved == "a":
            self.actor, = nets
        elif involved == "c":
            self.critic, = nets
        elif involved == "ac":
            self.actor, self.critic = nets

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(torch.load(checkpoint))
        self.actor.eval()

    def _get_desc_sort_order_a(self, state, candidates):
        proto_action = self.actor(state)
        proto_action = proto_action.unsqueeze(0)
        distances = torch.cdist(candidates, proto_action)
        distances = distances.reshape(-1)
        desc_sort_order = torch.argsort(distances, descending=False)
        return desc_sort_order

    def _get_desc_sort_order_c(self, state, candidates):
        state_repeated = state.repeat(len(candidates), 1)
        q_values = self.critic(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order

    def _get_desc_sort_order_ac(self, state, candidates):
        proto_action = self.actor(state, candidates)
        proto_action = proto_action.unsqueeze(0)
        distances = torch.cdist(candidates, proto_action)
        distances = distances.reshape(-1)
        desc_sort_order_actor = torch.argsort(distances, descending=False)
        k = round(0.5 * len(candidates))
        if k == 0:
            k = len(candidates)
        top_k_sort_order = desc_sort_order_actor[:k]
        critic_candidates = candidates[top_k_sort_order]

        state_repeated = state.repeat(len(critic_candidates), 1)
        q_values = self.critic(state_repeated, critic_candidates)
        q_values = q_values.reshape(-1)

        desc_sort_order_critic = torch.argsort(q_values, descending=True)
        desc_sort_order = torch.concatenate((
            top_k_sort_order[desc_sort_order_critic],
            desc_sort_order_actor[k:]
        ))
        return desc_sort_order
