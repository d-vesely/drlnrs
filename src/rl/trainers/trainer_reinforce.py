import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ..algorithms.reinforce import get_trainee, REINFORCE
from .trainer_base import _TrainerBase


class TrainerREINFORCE(_TrainerBase):
    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params,
                 ep_rm_path=None, seed=None):
        super().__init__(
            model_name,
            device,
            pos_rm_path,
            neg_rm_path,
            encoder_params,
            learning_params,
            model_params,
            ep_rm_path,
            seed
        )

    def set_trainee(self):
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
        )
        self.actor, = nets
        self.target_map = target_map
        self._print_num_params(self.actor.parameters(), name="actor")

        self.optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True,
            weight_decay=0.1
        )
        self.optimizers = [self.optimizer]
        self.schedulers = self._prepare_schedulers()
        self.REINFORCE = REINFORCE(self.device)

    def train_REINFORCE(self):
        n_steps = self.config_learning["n_steps"]
        gamma = self.config_learning["gamma"]
        freq_lr_schedule = self.config_learning["freq_lr_schedule"]
        freq_checkpoint_save = self.config_learning["freq_checkpoint_save"]

        self._print_init_lr()

        self.ckpt_num = 1
        running_hr = 0
        running_rr = 0
        for i in tqdm(range(n_steps)):
            batch = self.ep_rm_data[i]
            states, items, rewards = self._move_batch_to_device(batch)

            ep_len = len(states)
            n_clicks = rewards.sum()
            r_click = (ep_len - n_clicks) / n_clicks

            action_probs = self.actor(states, items)
            action = self.REINFORCE.act(action_probs)
            rs = np.zeros(ep_len)
            hits = 0
            for t in range(ep_len):
                if action[t] == 0:
                    if rewards[t] == 0:
                        rs[t] = 1
                    else:
                        rs[t] = -r_click
                else:
                    if rewards[t] == 0:
                        rs[t] = -1
                    else:
                        hits += 1
                        rs[t] = r_click

            self.REINFORCE.set_rewards_buffer(rs.copy())
            hit_rate = hits / (n_clicks.item() + self.REINFORCE.eps) * 100
            rec_rate = action.sum().item() / (n_clicks.item() + self.REINFORCE.eps) * 100
            running_hr = 0.05 * hit_rate + (1 - 0.05) * running_hr
            running_rr = 0.05 * rec_rate + (1 - 0.05) * running_rr

            # if (i % 100000) == 0:
            #     print(f"running rec rate: {running_rr}")
            #     print(f"running hit rate: {running_hr}")
            #     print(ep_len)
            #     print(
            #         f"items recommended {action.sum().item()}")
            #     print(f"items actually clicked {n_clicks.item()}")
            #     print(f"clicked items recommended {hits}")
            #     print(action_probs)

            returns = self.REINFORCE.get_returns(ep_len, gamma)
            policy_loss = self.REINFORCE.get_loss(returns)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            self.REINFORCE.reset()

            if (i + 1) % freq_lr_schedule == 0:
                self._update_learning_rate()

            if (i + 1) % freq_checkpoint_save == 0:
                self._save_model()
                self.ckpt_num += 1

        self._save_model(final=True)
