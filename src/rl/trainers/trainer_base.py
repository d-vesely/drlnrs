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
from ..replay_memory_dataset import ReplayMemoryDataset, ReplayMemoryEpisodicDataset, pad_candidates


class _TrainerBase():
    """Base trainer class"""

    def __init__(self, model_name, device,
                 pos_rm_path, neg_rm_path,
                 encoder_params, learning_params, model_params,
                 ep_rm_path, seed):
        """Initialize trainer

        Arguments:
            model_name -- name for directory
            device -- training device
            pos_rm_path -- path to positive experiences replay memory
            neg_rm_path -- path to negative experiences replay memory
            encoder_params -- encoder parameters
            learning_params -- learning parameters
            model_params -- model parameters
            ep_rm_path -- path to episodic replay memory
            seed -- random seed
        """
        # Set random seed for reproducibility
        if seed is not None:
            print(f"[INFO] setting seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Prepare training device
        self.device = device
        print(f"[INFO] device: {device}")

        # Prepare model directory
        model_dir = os.path.join(
            constants.MODELS_PATH,
            model_name
        )
        self.model_dir = model_dir
        print(f"[INFO] preparing directory {model_dir}")
        self._prepare_model_dir(seed)

        # Save configurations into config files
        self.config_encoder = encoder_params
        self.config_learning = learning_params
        self.config_model = model_params
        print(f"[INFO] writing config files to directory")
        self._write_config_files()

        # Prepare replay memory dataloaders
        if ep_rm_path is not None:
            print(f"[INFO] preparing episodic data and sampler")
            self.ep_rm_data = self._prepare_data_episodic(ep_rm_path)
        else:
            print(f"[INFO] preparing data and samplers")
            self.pos_rm_data, self.pos_sampler, \
                self.neg_rm_data, self.neg_sampler = self._prepare_data(
                    pos_rm_path,
                    neg_rm_path
                )

        print("[DONE] trainer initialized")

    def _prepare_model_dir(self, seed):
        """Prepare directory for model checkpoints, configs, results"""
        # Make model and configs directory
        self.config_dir = os.path.join(self.model_dir, "configs")
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

        # Make checkpoints directory according to random seed
        ckpt_dir_name = "checkpoints" if seed is None else f"checkpoints_{seed}"
        self.ckpt_dir = os.path.join(self.model_dir, ckpt_dir_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def _write_config_files(self):
        """Write parameters to config files"""
        # Write separate config file for encoder, learning parameters and
        # model parameters
        config_types = ["encoder", "learning", "model"]
        configs = [
            self.config_encoder,
            self.config_learning,
            self.config_model
        ]

        # Write config files in JSON format
        for i, config_type in enumerate(config_types):
            config_filepath = os.path.join(
                self.config_dir,
                f"config_{config_type}.json"
            )
            with open(config_filepath, "w") as config_file:
                json.dump(configs[i], config_file, indent=4)

    def _prepare_data(self, pos_rm_path, neg_rm_path):
        """Prepare positive and negative experiences replay memory"""
        returns = []
        for rm_path in [pos_rm_path, neg_rm_path]:
            # Get replay memory dataset
            rm_dataset = ReplayMemoryDataset(
                self.config_encoder["news_embedding_size"],
                rm_path,
                self.config_encoder,
                # use_ignore_history=True
            )
            # Create sampler for dataset
            # No replacement slows the sampling process enormously
            # We have enough data, this is not an issue
            sampler = RandomSampler(rm_dataset, replacement=True)
            returns.append(rm_dataset)
            returns.append(sampler)
        return returns

    def _prepare_data_episodic(self, rm_episodic_path):
        """Prepare episodic replay memory"""
        ep_rm_dataset = ReplayMemoryEpisodicDataset(
            rm_episodic_path,
            self.config_encoder,
        )
        return ep_rm_dataset

    def _prepare_dataloaders(self, batch_size):
        """Prepare dataloaders for positive and negative rm"""
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
        """Prepare learning rate scheduler for each optimizer"""
        # Load LR decay rate from parameters
        lr_decay_rate = self.config_learning["learning_decay_rate"]
        def lr_lambda(lr_step): return lr_decay_rate ** lr_step

        # Create identical scheduler for every optimizer
        schedulers = [
            optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            for optimizer in self.optimizers
        ]
        return schedulers

    def _sample_rm(self, pos_dataloader, pos_dl_iter,
                   neg_dataloader, neg_dl_iter):
        """Sample batch from replay memory, either positive or negative

        Arguments:
            pos_dataloader -- positive rm dataloader
            pos_dl_iter -- positive dataloader iterator
            neg_dataloader -- negative rm dataloader
            neg_dl_iter -- negative dataloader iterator

        Returns:
            Sampled batch.
        """
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
        """Print the number of trainable model parameters"""
        param_sum = sum(
            p.numel()
            for p in parameters
            if p.requires_grad
        )
        print(f"[INFO] number of trainable {name} parameters: {param_sum}")

    def _print_init_lr(self):
        """Print initial learning rate"""
        initial_lr = self.optimizers[0].param_groups[0]['lr']
        print(f"[INFO] initial learning rate: {initial_lr:.6f}")

    def _get_learning_params(self):
        """Get all learning parameters from config"""
        # Parameters that do not have to be adapted
        param_keys = [
            "gamma",
            "pos_mem_pref",
            "soft_target_update",
            "pos_mem_pref_adapt",
            "pos_mem_pref_adapt_step"
        ]
        # Parameters that have to be adapted to batch size
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
        # Adapt parameters to batch size where necessary
        batch_size = self.config_learning["batch_size"]
        adj_params = [
            self.config_learning[pk] // batch_size
            for pk in adj_param_keys
        ]
        # Adapt early checkpoint timesteps
        adj_progress_saves = [p // batch_size
                              for p in self.config_learning["progress_saves"]]
        params.extend(adj_params)
        params.append(adj_progress_saves)
        params.append(batch_size)
        return params

    def _move_batch_to_device(self, batch):
        """Move an entire batch of data to the device"""
        # Move all elements to device
        on_device_batch = []
        for elem in batch:
            on_device_batch.append(elem.to(self.device))

        # Return batch of moved elements
        return on_device_batch

    def _get_not_done(self, candidates):
        """Check if episode has ended"""
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
        """Check if a periodic event should occur

        Arguments:
            step_i -- current step
            soft_target_update -- whether to perform stu
            pos_mem_pref_adapt -- the pmp step frequency
            pos_mem_pref_adapt_step -- the pmp step size
            freq_target_update -- the target update frequency
            freq_lr_schedule -- the learning rate schedule frequency
            freq_checkpoint_save -- the checkpoint saving frequency
            freq_pos_mem_pref_adapt -- the pmp adaptation frequency
            progress_saves -- the early checkpoint timesteps

        Returns:
            Whether to print q-values for sanity checks
        """
        print_q = False
        # Update target network
        if (step_i + 1) % freq_target_update == 0:
            # Check if update should be soft or hard
            if soft_target_update:
                self._soft_update_target()
            else:
                self._update_target()

        # Update learning rate
        if (step_i + 1) % freq_lr_schedule == 0:
            self._update_learning_rate()

        # Adapt positive memory preference
        if (step_i + 1) % freq_pos_mem_pref_adapt == 0:
            # Step, if desired
            if pos_mem_pref_adapt:
                self.pos_mem_pref += pos_mem_pref_adapt_step
                print(
                    f"[INFO] positive memory preference raised to {self.pos_mem_pref}"
                )

        # Save checkpoint
        if (step_i + 1) % freq_checkpoint_save == 0 or \
                (step_i + 1) in progress_saves:
            # Save model, increase checkpoint counter, print q-values for sanity check
            self._save_model()
            self.ckpt_num += 1
            print_q = True
        return print_q

    def _update_target(self):
        """Hard update target network"""
        # Iterate over all online-target pairs
        for online_net, target_net in self.target_map.items():
            # Load entire state dict
            getattr(self, target_net).load_state_dict(
                getattr(self, online_net).state_dict()
            )

    def _soft_update_target(self):
        """Soft update target network"""
        # Load parameter tau
        tau = self.config_learning["tau"]

        # Iterate over all online-target pairs
        for online_net, target_net in self.target_map.items():
            target_sd = getattr(self, target_net).state_dict()
            online_sd = getattr(self, online_net).state_dict()
            # Soft update all values
            for key in online_sd:
                target_sd[key] = \
                    (online_sd[key] * tau) + \
                    (target_sd[key] * (1 - tau))

            getattr(self, target_net).load_state_dict(target_sd)

    def _update_learning_rate(self):
        """Update learning rate"""
        # Update LR for all schedulers
        for scheduler in self.schedulers:
            scheduler.step()
        # Print new LR
        new_lr = self.optimizers[0].param_groups[0]['lr']
        print(f"[INFO] new learning rate: {new_lr:.6f}")

    def _save_model(self, final=False):
        """Save checkpoint"""
        # Save final model
        if final:
            info_msg = f"[INFO] saving final model"
            model_name = f"final"
        # Save intermediary checkpoint
        else:
            info_msg = f"[INFO] saving model checkpoint {self.ckpt_num}"
            model_name = f"{self.ckpt_num}"
        print(info_msg)

        # Save pytorch nets
        for online_net in self.target_map.keys():
            torch.save(
                getattr(self, online_net).state_dict(),
                os.path.join(self.ckpt_dir, f"{online_net}_{model_name}.pth")
            )
        # Only in FQF
        if hasattr(self, "fpn"):
            torch.save(
                self.fpn.state_dict(),
                os.path.join(self.ckpt_dir, f"fpn_{model_name}.pth")
            )

    def update_config(self, config_name, update_dict):
        """Update configuration file"""
        # Replace k-v pair in config from update_dict
        for key, value in update_dict.items():
            getattr(self, config_name)[key] = value

        # Rewrite configs
        self._write_config_files()

    def train(self):
        """Main training function"""
        # Load all learning parameters
        gamma, pos_mem_pref, soft_target_update, \
            pos_mem_pref_adapt, pos_mem_pref_adapt_step, n_steps, \
            freq_target_update, freq_lr_schedule, freq_checkpoint_save, \
            freq_pos_mem_pref_adapt, progress_saves, batch_size \
            = self._get_learning_params()

        self._print_init_lr()

        self.pos_mem_pref = pos_mem_pref

        # Get dataloaders
        pos_dataloader, pos_dl_iter, \
            neg_dataloader, neg_dl_iter = self._prepare_dataloaders(batch_size)

        # Init checkpoint model number
        self.ckpt_num = 1
        print_q = False

        # Train
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

            # Single training step
            #! Must be implemented by specific algorithm's trainer class
            self._training_step(batch, i, gamma, print_q)

            # Check frequency events, unless this is the last step
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

        # Save final model
        self._save_model(final=True)
