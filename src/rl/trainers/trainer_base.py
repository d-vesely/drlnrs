import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from ... import constants
# from reinforce import ActorDiscrete, REINFORCE
from ..replay_memory_dataset import ReplayMemoryDataset, ReplayMemoryEpisodicDataset, pad_candidates


class _TrainerBase():
    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params,
                 ep_rm_path, seed):
        if seed is not None:
            print(f"[INFO] setting seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.device = device
        print(f"[INFO] device: {device}")

        model_dir = os.path.join(
            constants.MODELS_PATH,
            model_name
        )
        self.model_dir = model_dir

        print(f"[INFO] preparing directory {model_dir}")
        self._prepare_model_dir(seed)

        self.config_encoder = encoder_params
        self.config_learning = learning_params
        self.config_model = model_params

        print(f"[INFO] writing config files to directory")
        self._write_config_files()

        if ep_rm_path is not None:
            print(f"[INFO] preparing episodic data and sampler")
            self.ep_rm_data, self.sampler = \
                self._prepare_data_episodic(ep_rm_path)
        else:
            print(f"[INFO] preparing data and samplers")
            self.pos_rm_data, self.pos_sampler, \
                self.neg_rm_data, self.neg_sampler = self._prepare_data(
                    pos_rm_path,
                    neg_rm_path
                )

        print("[DONE] trainer initialized")

    def _prepare_model_dir(self, seed):
        self.config_dir = os.path.join(self.model_dir, "configs")
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        ckpt_dir_name = "checkpoints" if seed is None else f"checkpoints_{seed}"
        self.ckpt_dir = os.path.join(self.model_dir, ckpt_dir_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def _write_config_files(self):
        config_types = ["encoder", "learning", "model"]
        configs = [
            self.config_encoder,
            self.config_learning,
            self.config_model
        ]

        for i, config_type in enumerate(config_types):
            config_filepath = os.path.join(
                self.config_dir,
                f"config_{config_type}.json"
            )
            with open(config_filepath, "w") as config_file:
                json.dump(configs[i], config_file, indent=4)

    def _prepare_data(self, pos_rm_path, neg_rm_path):
        returns = []
        for rm_path in [pos_rm_path, neg_rm_path]:
            rm_dataset = ReplayMemoryDataset(
                self.config_encoder["news_embedding_size"],
                rm_path,
                self.config_encoder,
                # use_ignore_history=True
            )
            # Replacement slows the sampling process enormously
            # We have enough data, this is not an issue
            sampler = RandomSampler(rm_dataset, replacement=True)
            returns.append(rm_dataset)
            returns.append(sampler)
        return returns

    def _prepare_data_episodic(self, rm_episodic_path):
        ep_rm_dataset = ReplayMemoryEpisodicDataset(
            self.config_encoder["embeddings_map_paths"],
            rm_episodic_path,
            self.config_encoder,
        )
        sampler = SequentialSampler(ep_rm_dataset)  # TODO
        return ep_rm_dataset, sampler

    def _prepare_dataloaders(self, batch_size):
        returns = []
        for rm_data in [self.pos_rm_data, self.neg_rm_data]:
            dataloader = DataLoader(
                rm_data,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=pad_candidates
            )
            dl_iter = iter(dataloader)
            returns.append(dataloader)
            returns.append(dl_iter)
        return returns

    def _prepare_schedulers(self):
        lr_decay_rate = self.config_learning["learning_decay_rate"]
        def lr_lambda(lr_step): return lr_decay_rate ** lr_step

        schedulers = [
            optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            for optimizer in self.optimizers
        ]
        return schedulers

    def _sample_rm(self, pos_dataloader, pos_dl_iter,
                   neg_dataloader, neg_dl_iter):
        if np.random.uniform() < self.pos_mem_pref:
            try:
                batch = next(pos_dl_iter)
            except:
                pos_dl_iter = iter(pos_dataloader)
                batch = next(pos_dl_iter)
        else:
            try:
                batch = next(neg_dl_iter)
            except:
                neg_dl_iter = iter(neg_dataloader)
                batch = next(neg_dl_iter)
        return batch

    def _print_num_params(self, parameters, name=""):
        param_sum = sum(
            p.numel()
            for p in parameters
            if p.requires_grad
        )
        print(f"[INFO] number of trainable {name} parameters: {param_sum}")

    def _print_init_lr(self):
        initial_lr = self.optimizers[0].param_groups[0]['lr']
        print(f"[INFO] initial learning rate: {initial_lr:.6f}")

    def _get_learning_params(self):
        param_keys = [
            "gamma",
            "pos_mem_pref",
            "soft_target_update",
            "pos_mem_pref_adapt",
            "pos_mem_pref_adapt_step"
        ]
        adj_param_keys = [
            "n_steps",
            "freq_target_update",
            "freq_lr_schedule",
            "freq_checkpoint_save",
            "freq_pos_mem_pref_adapt"
        ]
        params = [
            self.config_learning[pk]
            for pk in param_keys
        ]
        batch_size = self.config_learning["batch_size"]
        adj_params = [
            self.config_learning[pk] // batch_size
            for pk in adj_param_keys
        ]
        adj_progress_saves = [p // batch_size
                              for p in self.config_learning["progress_saves"]]
        params.extend(adj_params)
        params.append(adj_progress_saves)
        params.append(batch_size)
        return params

    def _move_batch_to_device(self, batch):
        on_device_batch = []
        for elem in batch:
            on_device_batch.append(elem.to(self.device))
        return on_device_batch

    def _get_not_done(self, candidates):
        # The episode has ended if there are no more candidates
        # If first candidate contains NaN --> done
        not_done = torch.tensor([
            int(not torch.any(torch.isnan(c[0])))
            for c in candidates
        ])
        not_done = not_done.to(self.device)
        return not_done

    def _check_freq_events(self, step_i, soft_target_update,
                           pos_mem_pref_adapt, pos_mem_pref_adapt_step,
                           freq_target_update, freq_lr_schedule,
                           freq_checkpoint_save, freq_pos_mem_pref_adapt,
                           progress_saves):
        print_q = False
        if (step_i + 1) % freq_target_update == 0:
            if soft_target_update:
                self._soft_update_target()
            else:
                self._update_target()

        if (step_i + 1) % freq_lr_schedule == 0:
            self._update_learning_rate()

        if (step_i + 1) % freq_pos_mem_pref_adapt == 0:
            if pos_mem_pref_adapt:
                self.pos_mem_pref += pos_mem_pref_adapt_step
                print(
                    f"[INFO] positive memory preference raised to {self.pos_mem_pref}"
                )

        if (step_i + 1) % freq_checkpoint_save == 0 or \
                (step_i + 1) in progress_saves:
            self._save_model()
            self.ckpt_num += 1
            print_q = True
        return print_q

    def _update_target(self):
        for online_net, target_net in self.target_map.items():
            getattr(self, target_net).load_state_dict(
                getattr(self, online_net).state_dict()
            )

    def _soft_update_target(self):
        tau = self.config_learning["tau"]

        for online_net, target_net in self.target_map.items():
            target_sd = getattr(self, target_net).state_dict()
            online_sd = getattr(self, online_net).state_dict()
            for key in online_sd:
                target_sd[key] = \
                    (online_sd[key] * tau) + \
                    (target_sd[key] * (1 - tau))

            getattr(self, target_net).load_state_dict(target_sd)

    def _update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        new_lr = self.optimizers[0].param_groups[0]['lr']
        print(f"[INFO] new learning rate: {new_lr:.6f}")

    def _save_model(self, final=False):
        if final:
            info_msg = f"[INFO] saving final model"
            model_name = f"final"
        else:
            info_msg = f"[INFO] saving model checkpoint {self.ckpt_num}"
            model_name = f"{self.ckpt_num}"
        print(info_msg)
        for online_net in self.target_map.keys():
            torch.save(
                getattr(self, online_net).state_dict(),
                os.path.join(self.ckpt_dir, f"{online_net}_{model_name}.pth")
            )
        if hasattr(self, "fpn"):
            torch.save(
                self.fpn.state_dict(),
                os.path.join(self.ckpt_dir, f"fpn_{model_name}.pth")
            )

    def update_config(self, config_name, update_dict):
        for key, value in update_dict.items():
            getattr(self, config_name)[key] = value
        self._write_config_files()

    def set_trainee_REINFORCE(self):
        lr_decay_rate = self.config_learning["learning_decay_rate"]
        def lr_lambda(lr_step): return lr_decay_rate ** lr_step

        self.actor = ActorDiscrete(
            self.config_model["state_size"],
            self.config_model["item_size"],
            self.config_model["hidden_size"]
        ).to(self.device)

        self._print_num_params(self.actor.parameters(), name="actor")

        self.optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=self.config_learning["learning_rate"],
            amsgrad=True,
            weight_decay=0.1
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )

        self.REINFORCE = REINFORCE(self.device)

    def train_REINFORCE(self):
        gamma = self.config_learning["gamma"]
        pos_mem_pref = self.config_learning["pos_mem_pref"]
        n_steps = self.config_learning["n_steps"]

        freq_target_update = self.config_learning["freq_target_update"]
        freq_lr_schedule = self.config_learning["freq_lr_schedule"]
        freq_checkpoint_save = self.config_learning["freq_checkpoint_save"]
        soft_target_update = self.config_learning["soft_target_update"]

        initial_lr = self.optimizer.param_groups[0]['lr']
        print(f"[INFO] initial learning rate: {initial_lr:.6f}")

        # dataloader = DataLoader(self.episodic_memory,
        #                        shuffle=True, batch_size=32)
        # dl_iter = iter(dataloader)

        self.ckpt_num = 0
        running_hr = 0
        running_rr = 0
        for i in tqdm(range(1500000)):
            # Sample random step from replay memory
            # Choose whether to sample from positive or negative memory
            states, items, rewards = self.episodic_memory[i]

            # Move all elements to device
            states = states.to(self.device)
            items = items.to(self.device)
            rewards = rewards.to(self.device)

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
            hit_rate = hits / (n_clicks.item() + 0.0001) * 100
            rec_rate = action.sum().item() / (n_clicks.item() + 0.0001) * 100
            running_hr = 0.05 * hit_rate + (1 - 0.05) * running_hr
            running_rr = 0.05 * rec_rate + (1 - 0.05) * running_rr

            if (i % 100000) == 0:
                print(f"running rec rate: {running_rr}")
                print(f"running hit rate: {running_hr}")
                print(ep_len)
                print(
                    f"items recommended {action.sum().item()}")
                print(f"items actually clicked {n_clicks.item()}")
                print(f"clicked items recommended {hits}")
                print(action_probs)

            returns = self.REINFORCE.get_returns(ep_len)
            policy_loss = self.REINFORCE.get_loss(returns)

            self.optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.optimizer.step()

            self.REINFORCE.reset()

            if (i+1) % 250000 == 0:
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"[INFO] new learning rate: {new_lr:.6f}")

            if i % 250000 == 0:
                self._save_model(self.ckpt_num)
                self.ckpt_num += 1

        self._save_model(final=True)

    def train(self):
        gamma, pos_mem_pref, soft_target_update, \
            pos_mem_pref_adapt, pos_mem_pref_adapt_step, n_steps, \
            freq_target_update, freq_lr_schedule, freq_checkpoint_save, \
            freq_pos_mem_pref_adapt, progress_saves, batch_size \
            = self._get_learning_params()

        self._print_init_lr()

        self.pos_mem_pref = pos_mem_pref

        pos_dataloader, pos_dl_iter, \
            neg_dataloader, neg_dl_iter = self._prepare_dataloaders(batch_size)

        self.ckpt_num = 1
        print_q = False
        for i in tqdm(range(n_steps)):
            # Sample random step from replay memory
            batch = self._sample_rm(
                pos_dataloader,
                pos_dl_iter,
                neg_dataloader,
                neg_dl_iter
            )
            batch = self._move_batch_to_device(batch)
            batch.append(self._get_not_done(batch[-1]))

            self._training_step(batch, i, gamma, print_q)

            if i != (n_steps - 1):
                print_q = self._check_freq_events(
                    i,
                    soft_target_update,
                    pos_mem_pref_adapt,
                    pos_mem_pref_adapt_step,
                    freq_target_update,
                    freq_lr_schedule,
                    freq_checkpoint_save,
                    freq_pos_mem_pref_adapt,
                    progress_saves
                )

        self._save_model(final=True)
