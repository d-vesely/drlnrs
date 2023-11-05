import torch


class Encoder():
    """Wrapper for methods needed for state/action representation"""

    def __init__(self, embeddings_map_paths, news_enc_elements=["title"],
                 news_embedding_size=768, history_enc_method="mean",
                 weighted=False, alpha=0.99, history_max_len=None):
        """Initialize encoder

        Load embeddings and store configuration values for encoding
        functions.

        Arguments:
            embeddings_map_path -- dict containing paths to embeddings map files

        Keyword Arguments:
            news_enc_elements -- which news elements to encode (default: {["title"]})
            news_embedding_size -- size of embedding vector (default: {768})
            history_enc_method -- encoding method (default: {"mean"})
            weighted -- whether to apply weights to encoding method (default: {False})
            alpha -- weight (default: {0.99})
            history_max_len -- truncate histories to this length (default: {None})

        Raises:
            ValueError:
                - method value is not one of: ["stack", "mean", "dist", "ltstl"] OR
                - alpha not in interval (0, 1) OR
                - history_max_len not set for 'stack' method
        """
        # Check for potential value errors
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"[ERR] alpha must be in (0, 1), got {alpha}")

        avail_methods = ["stack", "mean", "dist", "ltstl"]
        if history_enc_method not in avail_methods:
            raise ValueError(f"[ERR] to_embed must be one of: {avail_methods}")

        if history_enc_method == "stack" and history_max_len is None:
            raise ValueError(
                f"[ERR] method 'stack' requires set history_max_len"
            )

        self.news_enc_elements = news_enc_elements
        self.news_embedding_size = news_embedding_size
        self._load_embeddings_maps(embeddings_map_paths)

        self.weighted = weighted
        self.history_max_len = history_max_len
        self.alpha = alpha
        self.history_enc_method = history_enc_method

    def _load_embeddings_maps(self, embeddings_map_paths):
        """Load embedding maps according to elements to be used"""
        assert list(embeddings_map_paths.keys()) == self.news_enc_elements
        self.embeddings_maps = {}
        for enc_elem in self.news_enc_elements:
            self.embeddings_maps[enc_elem] = torch.load(
                embeddings_map_paths[enc_elem]
            )

    def _get_empty_enc_history(self):
        """Get encoded history for empty history"""
        # Format of encoded history depends on method
        if self.history_enc_method == "stack":
            # stack of n embedding vectors
            enc_history = torch.zeros(
                (self.history_max_len, self.news_embedding_size))

        elif self.history_enc_method == "mean":
            # 1 mean embedding vector
            enc_history = torch.zeros(self.news_embedding_size)

        elif self.history_enc_method == "dist":
            # 3 concatenated embedding vectors
            enc_history = torch.zeros(3 * self.news_embedding_size)

        elif self.history_enc_method == "ltstl":
            # 3 concatenated embedding vectors
            enc_history = torch.zeros(3 * self.news_embedding_size)

        return enc_history

    def _get_history_cuts(self, history):
        """Split history into old and recent"""
        # Cut if wanted
        if self.history_max_len is not None:
            old_history = history[:-self.history_max_len]
            recent_history = history[-self.history_max_len:]
        # Entire history is recent history, no old history
        else:
            old_history = None
            recent_history = history.copy()
        return old_history, recent_history

    def _get_empty_stacks(self, recent_history_len, old_history_len=None):
        """Prepare empty stacks depending on method and max history length"""
        # Prepare stacks
        if self.history_max_len is not None and self.history_enc_method != "mean":
            # Old history has own stack
            if old_history_len == 0:
                old_history_stack = torch.zeros(self.news_embedding_size)
            else:
                old_history_stack = torch.empty(
                    (old_history_len, self.news_embedding_size)
                )

            # Empty slots in recent history are filled with 0
            recent_history_stack = torch.zeros(
                (self.history_max_len, self.news_embedding_size)
            )
        else:
            old_history_stack = None
            recent_history_stack = torch.empty(
                (recent_history_len, self.news_embedding_size)
            )

        return old_history_stack, recent_history_stack

    def _get_weights(self, history_len):
        """Prepare list of weights for weighted methods"""
        # Prepare list of weights
        # Newest news weighted by alpha^1
        # Oldest news weighted by alpha^(history_len)
        weight_list = [self.alpha ** (i+1) for i in range(history_len)]
        weights = torch.tensor(weight_list)
        weights = weights.flip(dims=(0,)).reshape(-1, 1)
        return weights

    def _get_mean_history_stack(self, history_stack, weights=None):
        """Compute mean over history stack"""
        history_stack_sum = torch.sum(history_stack, dim=0)
        if weights is not None:
            divisor = torch.sum(weights)
        else:
            divisor = history_stack.shape[0]
        enc_history = history_stack_sum / divisor
        return enc_history

    def _get_quantile_history_stack(self, history_stack):
        """Concatenate quartiles of history stack"""
        quantiles = torch.tensor([0.25, .5, .75])
        enc_history = torch.quantile(
            history_stack,
            q=quantiles,
            dim=0
        )
        enc_history = enc_history.reshape(-1)
        return enc_history

    def _get_ltstl_history_stack(self, history_stack, weights=None):
        """Create ltstl encoding of history stack"""
        # Get long-term encoding
        lt = self._get_mean_history_stack(
            history_stack,
            weights=weights
        )
        # Get short-term encoding
        if weights is not None:
            weights = weights[-5:]
        st = self._get_mean_history_stack(
            history_stack[-5:],
            weights=weights
        )
        # Get last read item
        l = history_stack[-1]
        enc_history = torch.cat((lt, st, l))
        return enc_history

    def _get_features(self, news_id):
        """Get numerical features for news"""
        return self.feature_map[news_id]

    def encode_history(self, history):
        """Encode reading history

        Arguments:
            history -- list of read news IDs

        Returns:
            encoded history as a torch tensor
        """
        # Handle empty histories by producing zero-tensor of correct size
        if len(history) == 0:
            enc_history = self._get_empty_enc_history()
            return enc_history

        # Cut off old history, if wanted
        old_history, recent_history = self._get_history_cuts(history)

        # Get lenghts, recent_history must exist
        old_history_len = None if old_history is None else len(old_history)
        recent_history_len = len(recent_history)

        # Prepare empty stacks
        old_history_stack, recent_history_stack = self._get_empty_stacks(
            recent_history_len,
            old_history_len
        )

        # Fill recent history stack
        stack_len = recent_history_stack.shape[0]
        # Iterate from latest to newest read
        for i, news_id in enumerate(recent_history):
            # Add embedding to stack
            recent_history_stack[i] = self.encode_news(news_id)

        # Fill old history stack, if needed
        if old_history_stack is not None:
            # Iterate over old news, order irrelevant
            for i, news_id in enumerate(old_history):
                # Add embedding to stack
                old_history_stack[i] = self.encode_news(news_id)

        # Apply weights to embeddings
        if self.weighted:
            weights = self._get_weights(stack_len)

            # Multiply embeddings by weights
            recent_history_stack = torch.mul(weights, recent_history_stack)

        # Leave stack untouched
        if self.history_enc_method == "stack":
            enc_recent_history = recent_history_stack

        # Get mean over all news in history
        elif self.history_enc_method == "mean":
            enc_history = self._get_mean_history_stack(
                recent_history_stack,
                weights=weights if self.weighted else None
            )

        elif self.history_enc_method == "dist":
            enc_history = self._get_quantile_history_stack(
                recent_history_stack
            )

        elif self.history_enc_method == "ltstl":
            enc_history = self._get_ltstl_history_stack(
                recent_history_stack,
                weights=weights if self.weighted else None
            )

        # This code
        if self.history_enc_method == "stack":
            if old_history_len != 0:
                enc_old_history = self._get_mean_history_stack(
                    old_history_stack,
                    weights=self._get_weights(old_history_len)
                )
            else:
                enc_old_history = old_history_stack

            # enc_history = torch.vstack(
            #    (enc_old_history, enc_recent_history))
            #! Currently, we do not append the old history to the
            #! stack, but the code above can be commented in
            enc_history = enc_recent_history

        return enc_history

    def encode_candidates(self, candidates):
        """Encode set of candidate news"""
        # Prepare stack
        candidates_stack = torch.empty(
            (len(candidates), self.news_embedding_size)
        )
        # Append each candidate's embedding to stack
        for i, news_id in enumerate(candidates):
            candidates_stack[i] = self.encode_news(news_id)
        return candidates_stack

    def encode_news(self, news_id):
        """Encode news item"""
        news_embedding = torch.empty(0)
        # Incrementally add embeddings of elements to be encoded
        for enc_elem in self.news_enc_elements:
            embedding = self.embeddings_maps[enc_elem][news_id]
            news_embedding = torch.cat((news_embedding, embedding))

        return news_embedding
