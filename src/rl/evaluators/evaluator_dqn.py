import torch

from ..algorithms.dqn import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorDQN(_EvaluatorBase):
    """Evaluator for DQN"""

    def __init__(self, development, model_name, device, seed=None, test_ckpt=None):
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
        # Set evaluation method according to model type
        if self.config_model["type"] == "dueling":
            self._get_desc_sort_order = self._get_desc_sort_order_duel
        else:
            self._get_desc_sort_order = self._get_desc_sort_order_regular

    def _load_checkpoint(self, checkpoint):
        """Load model checkpoint into net"""
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

    def _get_desc_sort_order_regular(self, state, candidates):
        """Get recommendation order from best to worst candidate"""
        # Repeat state for each candidate
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)

        # Get q-values and create sort order for candidates
        q_values = self.dqn(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order

    def _get_desc_sort_order_sequential(self, state, candidates, init_hist_len):
        """Get recommendation order from best to worst candidate, sequential approach"""
        if len(candidates) < 3:
            return self._get_desc_sort_order(state, candidates)
        state = state.unsqueeze(0)

        desc_sort_order = []
        q_values_full = torch.zeros(len(candidates), device=self.device)

        # Assume the first 3 recommendations are clicked
        for add in range(3):
            if len(state.shape) == 3:
                rep_shape = [len(candidates), 1, 1]
            else:
                rep_shape = [len(candidates), 1]
            state_repeated = state.repeat(*rep_shape)
            q_values = self.dqn(state_repeated, candidates)
            q_values = q_values.reshape(-1)
            # Make sure already recommended items are not reconsidered
            # We do not remove recommended items from the candidates, this is simpler
            for j in range(len(q_values_full)):
                if j in desc_sort_order:
                    q_values_full[j] = -10000
                else:
                    q_values_full[j] = q_values[j]

            # Find best candidate
            best_index = torch.argmax(q_values_full)
            desc_sort_order.append(best_index)
            best_news = candidates[best_index]
            # Assume candidate was clicked
            state = self._adjust_state(state, best_news, init_hist_len, add)

            state = state.to(self.device)
            candidates = candidates.to(self.device)

        # Recommend remaining items as usual
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        q_values = self.dqn(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        # Make sure the 3 already recommended items have very low q-values
        for j in range(len(q_values_full)):
            if j in desc_sort_order:
                q_values_full[j] = -10000
            else:
                q_values_full[j] = q_values[j]

        # Join recommendation orders
        desc_sort_order_rem = torch.argsort(q_values_full, descending=True)
        desc_sort_order_rem = desc_sort_order_rem[:-3]
        desc_sort_order.extend(desc_sort_order_rem)
        return torch.tensor(desc_sort_order)

    def _adjust_state(self, state, best_news, init_hist_len, add):
        """Adjust the state to newly clicked item

        This method is hardcoded for testing a sequential evaluation approach
        It expects/uses a weighted mean encoding with alpha = 0.999

        Arguments:
            state -- previous state
            best_news -- best candidate item
            init_hist_len -- initial history length
            add -- index of addition (add in range(3))

        Returns:
            adapted state representation that includes recommended item
        """

        # Recreate weights
        weight_list = [0.999 ** (i+1) for i in range(init_hist_len + add)]
        weights = torch.tensor(weight_list)
        divisor = torch.sum(weights)
        state *= divisor
        # Add new item and reweight entire history with alpha
        state += best_news
        state *= 0.999
        # Add new weight to divisor
        divisor += (0.999 ** (init_hist_len + add + 1))
        state /= divisor
        return state

    def _get_desc_sort_order_duel(self, state, candidates):
        """Get recommendation order from best to worst candidate for dueling DQN"""
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        vals, advs = self.dqn(state_repeated, candidates)
        q_values = vals.mean() + (advs - advs.mean())
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order
