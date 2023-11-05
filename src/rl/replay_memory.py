from collections import deque
from csv import writer
from io import StringIO
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm

from ..common_utils import read_pickled_data, read_feathered_data
from .news_rec_env import NewsRecEnv

tqdm.pandas()


class ReplayMemory():
    """Construct a replay memory of single steps in the RL problem

    The memory is an IO buffer, a csv writer writes to the buffer,
    allowing to incrementally construct the data. Once save() is called,
    the memory is pickled and the IO buffer closed.
    """

    def __init__(self, use_ignore_history):
        """Initialize the replay memory"""
        self.memory = StringIO()
        self.csv_writer = writer(self.memory)

        # Write header
        columns = [
            "recommended",
            "reward",
            "next_history",
            "next_candidates"
        ]
        if use_ignore_history:
            columns.append("next_ignore_history")
        self.csv_writer.writerow(columns)

    def add(self, recommended, obs, reward):
        """Add single row to replay memory

        Arguments:
            recommended -- news ID that was recommended
            obs -- observation dict (see news_rec_env.py)
            reward -- the obtained reward
        """
        row = [recommended, reward, obs["history"], obs["candidates"]]
        self.csv_writer.writerow(row)

    def add_impression(self, impression):
        """Add entire impression to replay memory

        Arguments:
            impression -- a list of lists, each list contains elements
            of the impression, i.e. timestamps, recommended items,
            rewards, next histories, next candidates, (next ignore histories)
        """
        impression_rows = zip(*impression)
        self.csv_writer.writerows(impression_rows)

    def save(self, save_dir, part_number, rm_dir_name, use_ignore_history):
        """Save replay memory

        Arguments:
            save_dir -- where to store the replay memory
            part_number -- replay memory is constructed in 
            parts due to memory constraints
            rm_dir_name -- subdirectory for replay memory
        """
        # Create directory within save_dir
        save_dir = os.path.join(save_dir, rm_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Seek to the start of the buffer and read it
        self.memory.seek(0)
        data_memory = pd.read_csv(self.memory)

        cols = ["next_history", "next_candidates"]
        if use_ignore_history:
            cols.append("next_ignore_history")

        # Convert strings back to lists
        for col in cols:
            data_memory[col] = data_memory[col].progress_apply(
                lambda x: [news_id[1:-1] for news_id in
                           x[1:-1].split(", ")]
            )
            data_memory[col] = data_memory[col].progress_apply(
                lambda x: [] if x == [""] else x
            )

        print(f"[INFO] saving part {part_number} to {save_dir}.")
        # Save the data
        data_memory.to_feather(os.path.join(
            save_dir, f"replay_memory_{part_number}.ftr"
        ))

        # Close the buffer
        self.memory.close()
        return data_memory


class ReplayMemoryBuilder():
    """Wrapper for building replay memory"""

    def __init__(self, path):
        """Initialize the replay memory builder with the path to the rm"""
        self.path = path

    def build(self, num_splits, use_ignore_history, rm_dir_name, behaviors_suffix):
        """Build the replay memory

        Arguments:
            num_splits -- in how many parts the rm should be built
            use_ignore_history -- whether to use the history of ignored items
            rm_dir_name -- the directory name for the rm
            behaviors_suffix -- the suffix of the behaviors data to load
        """
        # Load behaviors data
        print(
            f"[INFO] loading behaviors data from {self.path} ('behaviors{behaviors_suffix}')"
        )
        data_behaviors = read_pickled_data(
            [self.path, "preprocessed",
                f"behaviors{behaviors_suffix}.pkl"]
        )

        # Split data into equal parts
        split_indices = np.array_split(
            np.arange(len(data_behaviors)),
            num_splits
        )
        data_parts = [
            data_behaviors.iloc[indices]
            for indices in split_indices
        ]

        # Construct rm part by part
        for part_number, data in enumerate(data_parts):
            # Create replay memory and news recommendation simulator objects
            replay_memory = ReplayMemory(use_ignore_history)
            env = NewsRecEnv()

            print(f"[INFO] building part {part_number}")
            for row in tqdm(data.itertuples(), total=len(data)):
                # Load initial state
                timestamp = row.timestamp
                history = row.history
                candidates = set(row.shown_news)
                clicked_news = set(row.clicked_news)

                # Set initial state
                if use_ignore_history:
                    ignore_history = deque(row.ignore_history[:10], maxlen=10)
                    env.set_state(timestamp, history,
                                  candidates, ignore_history)
                else:
                    env.set_state(timestamp, history, candidates)

                # Simulate impression and add it to replay memory
                impression = env.simulate_impression(
                    clicked_news,
                    use_ignore_history=use_ignore_history
                )
                replay_memory.add_impression(impression)

                # Reset the environment
                env.reset()

            # Save the constructed part
            _ = replay_memory.save(
                self.path, part_number, rm_dir_name, use_ignore_history)
            print(f"[INFO] appending replay memory part {part_number}")

        print("[DONE] replay memory built")

    def concatenate(self, num_splits, rm_dir_name):
        """Concatenate individual parts of replay memory

        Arguments:
            num_splits -- in how many parts the rm should be built
            rm_dir_name -- the directory name for the rm
        """
        # Depending on RAM constraints, this function might have to be adapted
        # Concatenate all parts into full_replay_memory, or into two parts
        # Use commented out code as a starting point for concatenation into 2 parts

        # split_indices = np.array_split(np.arange(num_splits), 2)
        # for i in range(2):

        print("[INFO] reading replay memories")
        data_rm_parts = []
        # for j in split_indices[i]:

        # Read each part into a list
        for j in np.arange(num_splits):
            print(f"Appending {j}")
            data_rm_parts.append(
                read_feathered_data(
                    [self.path, rm_dir_name, f"replay_memory_{j}.ftr"])
            )

        # Concatenate all parts in the list into one
        print("[INFO] concatenating replay memories")
        full_replay_memory = pd.concat(data_rm_parts)
        full_replay_memory.reset_index(drop=True, inplace=True)

        # Save concatenated replay memory
        print("[INFO] saving full replay memory")
        full_replay_memory.to_feather(
            os.path.join(
                self.path,
                rm_dir_name,
                # f"full_replay_memory_{i}.ftr"
                f"full_replay_memory.ftr"
            )
        )
        print("[DONE] replay memory concatenated")

    def _read_full_replay_memory(self, rm_dir_name):
        """Read feathered replay memory"""
        full_replay_memory = read_feathered_data([
            self.path,
            rm_dir_name,
            "full_replay_memory.ftr"
        ])
        return full_replay_memory

    def extract_positive(self, rm_dir_name):
        """Extract positive experiences from replay memory"""
        # Load full replay memory
        print("[INFO] reading full replay memory")
        full_replay_memory = self._read_full_replay_memory(rm_dir_name)

        # Extract positive samples and save in dedicated feather
        print("[INFO] extracting positive samples")
        positive_samples = full_replay_memory[
            full_replay_memory["reward"] == 1
        ]
        positive_samples.reset_index(drop=True, inplace=True)
        print("[INFO] saving positive samples")
        positive_samples.to_feather(
            os.path.join(
                self.path,
                rm_dir_name,
                "positive_samples.ftr"
            )
        )
        print("[DONE] positive samples saved")

    def extract_negative(self, rm_dir_name, frac, random_state):
        """Extract negative experiences from replay memory

        Arguments:
            rm_dir_name -- the directory name for the rm
            frac -- the fraction of negative experiences to save
            random_state -- the seed for random sampling of
            negative experiences
        """
        # Load full replay memory
        print("[INFO] reading full replay memory")
        full_replay_memory = self._read_full_replay_memory(rm_dir_name)

        # Extract negative samples, by sampling a fraction of all
        # negative experiences, and save in dedicated feather
        print("[INFO] extracting negative samples")
        negative_samples = full_replay_memory[
            full_replay_memory["reward"] == 0
        ].sample(
            frac=frac,
            random_state=random_state
        )
        negative_samples.reset_index(drop=True, inplace=True)
        print("[INFO] saving negative samples")
        negative_samples.to_feather(
            os.path.join(
                self.path,
                rm_dir_name,
                f"negative_samples_{int(frac*100)}.ftr"
            )
        )
        print("[DONE] negative samples saved")


class ReplayMemoryEpisodicBuilder():
    """Wrapper for building episodic replay memory"""

    def __init__(self, path):
        self.path = path

    def build(self):
        """Build episodic replay memory"""
        # Load behaviors data from given path
        print(f"[INFO] loading behaviors data from {self.path}")
        data_behaviors = read_pickled_data(
            [self.path, "preprocessed", "behaviors.pkl"]
        )

        # Prepare simulation environment
        env = NewsRecEnv()
        rows = []

        for row in tqdm(data_behaviors.itertuples(), total=len(data_behaviors)):
            # Load initial state
            history = row.history
            states = [history.copy()]
            candidates = set(row.shown_news)
            clicked_news = set(row.clicked_news)

            # Set initial state
            env.set_state(history, candidates)

            # Simulate entire impression
            impression = env.simulate_impression(clicked_news)

            # Save all states (except final state), items and rewards
            states.extend(impression[2][:-1])
            items = impression[0]
            rewards = impression[1]
            rows.append((states, items, rewards))

            # Reset the environment
            env.reset()

        # Save episodic rm
        data_rm_episodic = pd.DataFrame(
            rows, columns=["states", "items", "rewards"]
        )
        save_dir = os.path.join(self.path, "replay_memory_episodic")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] saving episodic replay memory")
        data_rm_episodic.to_feather(os.path.join(
            save_dir, f"replay_memory_episodic.ftr"
        ))

        return data_rm_episodic
