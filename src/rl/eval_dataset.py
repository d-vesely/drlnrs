import pandas as pd
from torch.utils.data import Dataset

from .encoder import Encoder


class EvalDataset(Dataset):
    """Torch Dataset wrapper for evaluation data"""

    def __init__(self, eval_data_path, encoder_params={}, development=False):
        """Initialize evaluation dataset

        Arguments:
            eval_data_path -- path to evaluation data

        Keyword Arguments:
            encoder_params -- encoder parameters/kwargs (default: {{}})
            development -- whether this dataset is for dev data or not (default: {False})
        """
        self.encoder = Encoder(**encoder_params)
        self.eval_data = pd.read_pickle(eval_data_path)
        self.development = development

    def __len__(self):
        """Return dataset length"""
        return len(self.eval_data)

    def __getitem__(self, index):
        """Return item at index"""
        # Load data depending on dev or test data
        if self.development:
            # Ignore user_id, timestamp and ignored_news
            impression_id, _, history, _, clicked_news, _, candidates \
                = self.eval_data.iloc[index]
        else:
            # Ignore user_id and timestamp
            impression_id, _, history, _, candidates \
                = self.eval_data.iloc[index]

        # Encode history and candidates
        enc_history = self.encoder.encode_history(history)
        enc_candidates = self.encoder.encode_candidates(candidates)

        # Return values depending on dev or test data
        if self.development:
            return impression_id, enc_history, enc_candidates, clicked_news, candidates, len(history)

        return impression_id, enc_history, enc_candidates
