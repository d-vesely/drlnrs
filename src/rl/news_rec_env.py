from collections import deque
import random


class NewsRecEnv():
    """News recommendation simulation environment

    # TODO
    """

    def __init__(self, timestamp=None, history=[], candidates=set(), ignore_history=deque(maxlen=10)):
        """Initialize news rec environment

        Keyword Arguments:
            history -- list of previously read news (default: {[]})
            candidates -- list of candidates available for 
            recommendation (default: {set()})
        """
        self.timestamp = timestamp
        self.history = history
        self.candidates = candidates
        self.ignore_history = ignore_history

    def set_state(self, timestamp, history, candidates, ignore_history=deque(maxlen=10)):
        """Set environment's state"""
        self.timestamp = timestamp
        self.history = history
        self.candidates = candidates
        self.ignore_history = ignore_history

    def _get_obs(self):
        """Get observation of current state"""
        obs = {
            "timestamp": self.timestamp,
            "history": self.history.copy(),
            "candidates": self.candidates.copy(),
            "ignore_history": self.ignore_history.copy()
        }
        return obs

    def sample(self):
        """Sample random news to recommend"""
        return random.choice(list(self.candidates))

    def reset(self):
        """Reset the environment"""
        self.timestamp = None
        self.history = []
        self.candidates = set()
        self.ignore_history = deque(maxlen=10)
        obs = self._get_obs()
        return obs

    def step(self, recommended, clicked):
        """Perform single step in environment

        Arguments:
            recommended -- news ID that was recommended
            clicked -- whether the recommended news was read or not

        Returns:
            obs -- observation of the new state
            reward -- the numeric reward
            done -- whether the episode has ended
        """
        reward = 0
        done = False

        # Remove recommended news from candidate set
        self.candidates.remove(recommended)
        # Episode ends when no more candidates are available
        if len(self.candidates) == 0:
            done = True

        # If news was read, add reward and append news to reading history
        if clicked:
            reward = 1
            self.history.append(recommended)

        # Get observation of new state
        obs = self._get_obs()

        return obs, reward, done

    def simulate_impression(self, clicked_news, use_ignore_history=False):
        """Simulate entire impression

        Instead of performing a single step, this method runs an entire
        impression and returns information about each step.

        Arguments:
            clicked_news -- set of clicked news in the impression

        Returns:
            a list of lists, with each list containing information
            about each step of the impression
        """
        # Randomly shuffle the candidates
        # Instead of sampling, we can iterate over candidates
        candidates_list = list(self.candidates)
        random.shuffle(candidates_list)

        # Prepare lists containing information about each step
        recommendeds = []
        rewards = []
        next_candidates = []
        next_histories = []
        if use_ignore_history:
            next_ignore_histories = []

        # Iteratively recommend news from the shuffled candidates
        for i, c in enumerate(candidates_list):
            recommendeds.append(c)
            reward = 0

            # Check if recommended news was read
            if c in clicked_news:
                reward = 1
                self.history.append(c)
            elif use_ignore_history:
                self.ignore_history.append(c)

            rewards.append(reward)
            # All other candidates are next candidates
            next_candidates.append(candidates_list[i+1:].copy())
            next_histories.append(self.history.copy())
            if use_ignore_history:
                next_ignore_histories.append(list(self.ignore_history.copy()))

        timestamps = [self.timestamp] * len(recommendeds)
        # Return list of lists
        impression = [timestamps, recommendeds,
                      rewards, next_histories, next_candidates]
        if use_ignore_history:
            impression.append(next_ignore_histories)
        return impression

    def __repr__(self):
        """Write out current state"""
        return f"Timestamp: {self.timestamp}\nHistory: {self.history}\nCandidates: {self.candidates}"
