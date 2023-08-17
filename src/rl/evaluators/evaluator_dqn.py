import torch

from ..algorithms.dqn import get_evaluatee
from .evaluator_base import _EvaluatorBase


class EvaluatorDQN(_EvaluatorBase):
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
        self.dqn, = nets
        if self.config_model["type"] == "dueling":
            self._get_desc_sort_order = self._get_desc_sort_order_duel
        else:
            self._get_desc_sort_order = self._get_desc_sort_order_regular

    def _load_checkpoint(self, checkpoint):
        self.dqn.load_state_dict(torch.load(checkpoint))
        self.dqn.eval()

    def _get_desc_sort_order_regular(self, state, candidates):
        state = state.unsqueeze(0)
        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        q_values = self.dqn(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        desc_sort_order = torch.argsort(q_values, descending=True)
        return desc_sort_order

    def _get_desc_sort_order_sequential(self, state, candidates, init_hist_len):
        if len(candidates) < 3:
            return self._get_desc_sort_order(state, candidates)
        state = state.unsqueeze(0)

        desc_sort_order = []
        q_values_full = torch.zeros(len(candidates), device=self.device)

        for add in range(3):
            if len(state.shape) == 3:
                rep_shape = [len(candidates), 1, 1]
            else:
                rep_shape = [len(candidates), 1]
            state_repeated = state.repeat(*rep_shape)
            q_values = self.dqn(state_repeated, candidates)
            q_values = q_values.reshape(-1)
            for j in range(len(q_values_full)):
                if j in desc_sort_order:
                    q_values_full[j] = -10000
                else:
                    q_values_full[j] = q_values[j]

            best_index = torch.argmax(q_values_full)
            desc_sort_order.append(best_index)
            best_news = candidates[best_index]
            state = self._adjust_state(state, best_news, init_hist_len, add)

            state = state.to(self.device)
            candidates = candidates.to(self.device)

        if len(state.shape) == 3:
            rep_shape = [len(candidates), 1, 1]
        else:
            rep_shape = [len(candidates), 1]
        state_repeated = state.repeat(*rep_shape)
        q_values = self.dqn(state_repeated, candidates)
        q_values = q_values.reshape(-1)
        for j in range(len(q_values_full)):
            if j in desc_sort_order:
                q_values_full[j] = -10000
            else:
                q_values_full[j] = q_values[j]

        desc_sort_order_rem = torch.argsort(q_values_full, descending=True)
        desc_sort_order_rem = desc_sort_order_rem[:-3]
        desc_sort_order.extend(desc_sort_order_rem)
        return torch.tensor(desc_sort_order)

    def _adjust_state(self, state, best_news, init_hist_len, add):
        weight_list = [0.999 ** (i+1) for i in range(init_hist_len + add)]
        weights = torch.tensor(weight_list)
        divisor = torch.sum(weights)
        state *= divisor
        state += best_news
        state *= 0.999
        divisor += (0.999 ** (init_hist_len + add + 1))
        state /= divisor
        return state

    def _get_desc_sort_order_duel(self, state, candidates):
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
