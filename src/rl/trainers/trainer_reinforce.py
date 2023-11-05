import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ..algorithms.reinforce import get_trainee, REINFORCE
from .trainer_base import _TrainerBase


class TrainerREINFORCE(_TrainerBase):
    """Trainer for REINFORCE"""

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
        """Prepare trainee"""
        # Get model architecture and online-target map
        nets, target_map = get_trainee(
            self.config_model,
            self.device,
        )
        self.actor, = nets
        self.target_map = target_map

        # Print number of trainable parameters
        self._print_num_params(self.actor.parameters(), name="actor")

        # Prepare optimizers and schedulers
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
        """REINFORCE training algorithm"""
        # Unlike for all other algorithms, this method is not a single step,
        # but rather the entire algorithm
        # Load relevant hyperparameters
        n_steps = self.config_learning["n_steps"]
        gamma = self.config_learning["gamma"]
        freq_lr_schedule = self.config_learning["freq_lr_schedule"]
        freq_checkpoint_save = self.config_learning["freq_checkpoint_save"]

        # Print initial learning rate
        self._print_init_lr()

        # Init checkpoint model number
        self.ckpt_num = 1
        running_hr = 0
        running_rr = 0

        # Train
        for i in tqdm(range(n_steps)):
            # Sample random step from episodic replay memory
            batch = self.ep_rm_data[i]
            states, items, rewards = self._move_batch_to_device(batch)

            # Determine episode length
            ep_len = len(states)
            # The reward must be adapted to the total amount of
            # clicks and non-clicks, otherwise, the agent will always learn
            # to ONLY recommend or ONLY ignore candidates
            n_clicks = rewards.sum()
            r_click = (ep_len - n_clicks) / n_clicks

            # Get action probabilities and select action
            action_probs = self.actor(states, items)
            action = self.REINFORCE.act(action_probs)

            rs = np.zeros(ep_len)
            hits = 0
            # Reward distribution is balanced, such that only taking the
            # same action every time will yield reward 0 on average
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

            # Save rewards in buffer
            self.REINFORCE.set_rewards_buffer(rs.copy())

            # Compute monitoring stats
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

            # Get returns and compute policy loss
            returns = self.REINFORCE.get_returns(ep_len, gamma)
            policy_loss = self.REINFORCE.get_loss(returns)

            # Optimize
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            self.REINFORCE.reset()

            # Check frequency events
            if (i + 1) % freq_lr_schedule == 0:
                self._update_learning_rate()

            if (i + 1) % freq_checkpoint_save == 0:
                self._save_model()
                self.ckpt_num += 1

        # Save final model
        self._save_model(final=True)
