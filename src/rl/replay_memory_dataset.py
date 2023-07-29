import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .encoder import Encoder


class ReplayMemoryDataset(Dataset):
    """Torch Dataset wrapper for replay memory"""

    def __init__(self, news_embedding_size, replay_memory_path,
                 encoder_kwargs={}, use_ignore_history=False):
        """Initialize replay memory dataset

        Arguments:
            embeddings_map_path -- path to embeddings map
            replay_memory_path -- path to replay memory pickle

        Keyword Arguments:
            encoder_kwargs -- encoder_kwargs (default: {{}})
        """
        # Initialize encoder
        assert news_embedding_size == 768 or news_embedding_size == encoder_kwargs[
            "news_embedding_size"]
        self.encoder = Encoder(**encoder_kwargs)
        self.news_embedding_size = news_embedding_size
        # Read replay memory pandas dataframe
        self.replay_memory = pd.read_feather(replay_memory_path)
        self.use_ignore_history = use_ignore_history

    def __len__(self):
        """Return dataset length"""
        return len(self.replay_memory)

    def __getitem__(self, index):
        """Return item at index"""
        # Fetch column elements
        if self.use_ignore_history:
            # timestamp, recommended, reward, next_history, next_candidates, next_ignore_history = self.replay_memory.iloc[
            #     index]
            recommended, reward, next_history, next_candidates, next_ignore_history = self.replay_memory.iloc[
                index]
            next_ignore_history = next_ignore_history.tolist()
        else:
            # timestamp, recommended, reward, next_history, next_candidates = self.replay_memory.iloc[
            #     index]
            recommended, reward, next_history, next_candidates = self.replay_memory.iloc[
                index]
        next_history = next_history.tolist()

        # timestamp = timestamp.to_datetime64().view(np.int64) / 3.6e12

        # Current history is the next history
        # If the recommended news was read, remove last element from current history
        history = next_history.copy()
        if reward == 1:
            history.pop()

        if self.use_ignore_history:
            ignore_history = next_ignore_history.copy()
            if reward == 0:
                ignore_history.pop()

        # Encode current and next history
        enc_history = self.encoder.encode_history(history)
        enc_next_history = self.encoder.encode_history(next_history)
        if self.use_ignore_history:
            enc_ignore_history = self.encoder.encode_history(ignore_history)
            enc_next_ignore_history = self.encoder.encode_history(
                next_ignore_history)

        # If no more candidates, set to NaN vector
        if len(next_candidates) == 0:
            enc_next_candidates = torch.full(
                (1, self.news_embedding_size), float('nan'))
        else:
            enc_next_candidates = self.encoder.encode_candidates(
                next_candidates
            )

        # Get embedding of recommended news
        enc_recommended = self.encoder.encode_news(recommended)

        # Convert reward to tensor
        reward_tensor = torch.tensor(reward)
        if reward == 0:
            reward_tensor -= 1

        if self.use_ignore_history:
            return enc_history, enc_recommended, reward_tensor, enc_next_history, enc_next_candidates, enc_ignore_history, enc_next_ignore_history
        # return timestamp, enc_history, enc_recommended, reward_tensor, enc_next_history, enc_next_candidates
        return enc_history, enc_recommended, reward_tensor, enc_next_history, enc_next_candidates


def pad_candidates(batch):
    b_state = torch.stack([data[0] for data in batch])
    b_action = torch.stack([data[1] for data in batch])
    b_reward = torch.stack([data[2] for data in batch])
    b_next_state = torch.stack([data[3] for data in batch])
    b_candidates = [
        data[4]
        for data in batch
    ]
    # b_ignore_history = torch.stack([data[5] for data in batch])
    # b_next_ignore_history = torch.stack([data[6] for data in batch])

    padded_candidates = pad_sequence(
        b_candidates,
        batch_first=True,
        padding_value=-torch.inf
    )

    # , b_ignore_history, b_next_ignore_history]
    return [b_state, b_action, b_reward, b_next_state, padded_candidates]


class ReplayMemoryEpisodicDataset(Dataset):
    def __init__(self, replay_memory_path, encoder_kwargs={}):
        """Initialize replay memory dataset

        Arguments:
            embeddings_map_path -- path to embeddings map
            replay_memory_path -- path to replay memory pickle

        Keyword Arguments:
            encoder_kwargs -- encoder_kwargs (default: {{}})
        """
        # Initialize encoder
        self.encoder = Encoder(**encoder_kwargs)
        # Read replay memory pandas dataframe
        self.replay_memory_episodic = pd.read_feather(replay_memory_path)

    def __len__(self):
        """Return dataset length"""
        return len(self.replay_memory_episodic)

    def __getitem__(self, index):
        """Return item at index"""
        # Fetch column elements
        states, items, rewards = self.replay_memory_episodic.iloc[index]

        enc_states = []
        for history in states:
            enc_history = self.encoder.encode_history(history)
            enc_states.append(enc_history)

        enc_items = []
        for item in items:
            # Get embedding of recommended news
            enc_recommended = self.encoder.encode_news(item)
            enc_items.append(enc_recommended)

        enc_states = torch.stack(enc_states)
        enc_items = torch.stack(enc_items)
        enc_rewards = torch.stack([torch.tensor(r) for r in rewards])

        return enc_states, enc_items, enc_rewards
