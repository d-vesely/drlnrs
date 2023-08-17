from csv import writer
from io import StringIO
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm

from ... import constants
from ..eval_dataset import EvalDataset

tqdm.pandas()


class _EvaluatorBase():
    def __init__(self, development, model_name, device, seed, test_ckpt):
        self.development = development
        self.eval_path = os.path.join(
            constants.DEV_PATH if development else constants.TEST_PATH,
            "preprocessed",
            "behaviors.pkl"
        )
        self.model_name = model_name

        self.device = device
        print(f"[INFO] device: {device}")

        model_dir = os.path.join(
            constants.MODELS_PATH,
            model_name
        )
        self.model_dir = model_dir
        ckpt_dir_name = "checkpoints" if seed is None else f"checkpoints_{seed}"
        ckpts_dir = os.path.join(model_dir, ckpt_dir_name)

        print("[INFO] preparing predictions directory")
        self._prepare_pred_dir(seed)

        print(f"[INFO] reading config files")
        self._read_config_files()

        if development:
            self.model_ckpts = [
                os.path.join(ckpts_dir, filename.name)
                for filename in os.scandir(ckpts_dir)
            ]
            ckpts = [os.path.basename(ckpt) for ckpt in self.model_ckpts]
            if hasattr(self, "ac"):
                mid = len(ckpts) // 2
                self.model_ckpts = [(self.model_ckpts[i], self.model_ckpts[i + mid])
                                    for i in range(mid)]
            print(f"[INFO] model checkpoints: {ckpts}")
        else:
            self.model_ckpts = [
                os.path.join(ckpts_dir, f"{test_ckpt}.pth")]
            print(f"[INFO] model checkpoint: {self.model_ckpts[0]}")

            self.test_dir = os.path.join(
                self.pred_dir,
                f"test_{test_ckpt}"
            )
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)

        print(f"[INFO] preparing data and sampler")
        self.eval_data, self.sampler = self._prepare_data()

        print("[DONE] evaluator initialized")

    def _read_config_files(self):
        config_types = ["encoder", "model"]
        configs = []
        for config_type in config_types:
            config_filepath = os.path.join(
                self.model_dir,
                "configs",
                f"config_{config_type}.json"
            )
            with open(config_filepath, "r") as config_file:
                configs.append(json.load(config_file))

        self.config_encoder, self.config_model = configs

    def _prepare_pred_dir(self, seed):
        pred_dir_name = "predictions" if seed is None else f"predictions_{seed}"
        self.pred_dir = os.path.join(
            self.model_dir,
            pred_dir_name
        )
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

    def _prepare_data(self):
        eval_data = EvalDataset(
            self.eval_path,
            encoder_params=self.config_encoder,
            development=self.development
        )
        sampler = SequentialSampler(eval_data)
        return eval_data, sampler

    def _prepare_prediction_buffer(self):
        self.results = StringIO()
        csv_writer = writer(self.results)
        if self.development:
            columns = ["checkpoint", "mean_return"]
        else:
            columns = ["impression_id", "ranking"]
        csv_writer.writerow(columns)
        return csv_writer

    def _write_predictions_file(self):
        print(f"[INFO] writing predictions file to {self.test_dir}")
        # Read csv buffer into pandas dataframe
        self.results.seek(0)
        data_predictions = pd.read_csv(self.results)
        # Convert strings back to lists
        data_predictions["ranking"] = data_predictions["ranking"].progress_apply(
            lambda x: f"[{','.join(x[1:-1].split())}]"
        )
        # Write to txt file
        data_predictions.to_csv(
            os.path.join(self.test_dir, "prediction.txt"),
            sep=' ',
            index=False,
            header=False
        )

    def _write_eval_result_file(self, suffix=""):
        print(
            f"[INFO] writing evaluation results file to {self.pred_dir}"
        )
        # Read csv buffer into pandas dataframe
        self.results.seek(0)
        data_eval_results = pd.read_csv(self.results)
        # Write to txt file
        data_eval_results.to_csv(
            os.path.join(self.pred_dir, f"eval_results{suffix}.txt"),
            sep='\t',
            index=False,
            header=True
        )

    def _get_return(self, desc_sort_order, clicked_news_list, shown_news_list):
        # Compute return (discounted sum of rewards)
        clicked_news = set(clicked_news_list)

        # Order shown news from best to worst
        shown_news = np.array(shown_news_list)
        shown_news = shown_news[desc_sort_order.cpu().numpy()]

        G = 0
        # Discount value must be < 1
        # Specific value does not matter
        # but should remain constant over experiments, otherwise
        # returns cannot be compared
        gamma = 0.9
        for t, news_id in enumerate(shown_news):
            reward = 0
            if news_id in clicked_news:
                reward = 1
            G += ((gamma**t) * reward)
        return G

    def _get_prediction(self, desc_sort_order, impression_id):
        # Create ranking
        # Best q-value --> rank 1
        ranking = np.zeros(len(desc_sort_order), dtype=np.uintc)
        for i, idx in enumerate(desc_sort_order):
            # Add 1 to ranking (smallest rank is 1)
            ranking[idx] = i+1

        # Write prediction to buffer
        pred = [impression_id, ranking]
        return pred

    def _compute_mean_return(self, returns):
        mean_return = np.array(returns).mean()
        print(f"[RESULT] Return: {mean_return:.4f}")
        return mean_return

    def evaluate(self):
        csv_writer = self._prepare_prediction_buffer()
        for ckpt in self.model_ckpts:
            if hasattr(self, "ac"):
                ckpt_name = ckpt
            else:
                ckpt_name = os.path.basename(ckpt)
            print(
                f"[INFO] evaluating '{self.model_name}', checkpoint '{ckpt_name}'"
            )

            self._load_checkpoint(ckpt)

            if self.development:
                returns = []

            for index in tqdm(self.sampler):
                if self.development:
                    impression_id, state, candidates, \
                        clicked_news, shown_news, _ = self.eval_data[index]
                else:
                    impression_id, state, candidates = self.eval_data[index]

                # Move all elements to device
                state = state.to(self.device)
                candidates = candidates.to(self.device)

                with torch.no_grad():
                    desc_sort_order = self._get_desc_sort_order(
                        state,
                        candidates
                    )

                if self.development:
                    G = self._get_return(
                        desc_sort_order,
                        clicked_news,
                        shown_news
                    )
                    returns.append(G)
                else:
                    pred = self._get_prediction(desc_sort_order, impression_id)
                    csv_writer.writerow(pred)

            if self.development:
                mean_return = self._compute_mean_return(returns)
                csv_writer.writerow([ckpt_name, mean_return])
            else:
                self._write_predictions_file()

        if self.development:
            self._write_eval_result_file()

        print("[DONE] evaluation completed")

    def evaluate_sequential(self):
        csv_writer = self._prepare_prediction_buffer()
        for ckpt in self.model_ckpts:
            if hasattr(self, "ac"):
                ckpt_name = ckpt
            else:
                ckpt_name = os.path.basename(ckpt)

            if "final" not in ckpt_name:
                continue
            print(
                f"[INFO] evaluating '{self.model_name}', checkpoint '{ckpt_name}'"
            )

            self._load_checkpoint(ckpt)

            if self.development:
                returns = []

            for index in tqdm(self.sampler):
                if self.development:
                    impression_id, state, candidates, \
                        clicked_news, shown_news, init_hist_len = self.eval_data[index]
                else:
                    impression_id, state, candidates = self.eval_data[index]

                # Move all elements to device
                state = state.to(self.device)
                candidates = candidates.to(self.device)

                with torch.no_grad():
                    desc_sort_order = self._get_desc_sort_order_sequential(
                        state,
                        candidates,
                        init_hist_len
                    )

                if self.development:
                    G = self._get_return(
                        desc_sort_order,
                        clicked_news,
                        shown_news
                    )
                    returns.append(G)
                else:
                    pred = self._get_prediction(desc_sort_order, impression_id)
                    csv_writer.writerow(pred)

            if self.development:
                mean_return = self._compute_mean_return(returns)
                csv_writer.writerow([ckpt_name, mean_return])
            else:
                self._write_predictions_file()

        if self.development:
            self._write_eval_result_file(suffix="_seq")

        print("[DONE] evaluation completed")
