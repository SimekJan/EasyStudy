import numpy as np
from abc import ABC
from collections import defaultdict

import torch
import torch.nn as nn

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

class NCF(AlgorithmBase, ABC):

    def __init__(
        self,
        loader,
        positive_threshold,
        emb_dim=16,
        lr=1e-3,
        epochs=5,
        device=None,
        **kwargs
    ):
        print("_____INITIALIZING_MODEL_____", flush=True)

        self._loader = loader
        self._ratings_df = loader.ratings_df

        self._users = self._ratings_df.user.unique()
        self._items = self._ratings_df.item.unique()

        self.user2idx = {u: i for i, u in enumerate(self._users)}
        self.item2idx = {i: j for j, i in enumerate(self._items)}
        self.idx2item = {v: k for k, v in self.item2idx.items()}

        self._num_users = len(self._users)
        self._num_items = len(self._items)

        self._threshold = positive_threshold
        self._emb_dim = emb_dim
        self._lr = lr
        self._epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._build_model()
        self._prepare_data()

    # ---------------- MODEL ----------------
    def _build_model(self):

        print("_____BUILDING_MODEL_____", flush=True)

        self.gmf_item_emb = nn.Embedding(self._num_items, self._emb_dim)
        self.mlp_item_emb = nn.Embedding(self._num_items, self._emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self._emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.optimizer = torch.optim.Adam(
            list(self.gmf_item_emb.parameters()) +
            list(self.mlp_item_emb.parameters()) +
            list(self.mlp.parameters()),
            lr=self._lr
        )

    # ---------------- DATA ----------------
    def _prepare_data(self):

        print("_____PREPARING_DATA_____", flush=True)

        self.pos_interactions = []
        self.user_history = defaultdict(list)

        df = self._ratings_df[["user", "item", "rating"]]

        for u_raw, i_raw, r in df.itertuples(index=False):

            if u_raw not in self.user2idx or i_raw not in self.item2idx:
                continue

            if r >= self._threshold:
                u = self.user2idx[u_raw]
                i = self.item2idx[i_raw]

                self.pos_interactions.append((u, i))
                self.user_history[u].append(i)

        # popularity
        self.item_popularity = np.zeros(self._num_items)

        for _, i in self.pos_interactions:
            self.item_popularity[i] += 1

        # smoothing
        self.item_popularity = np.power(self.item_popularity, 0.75)
        self.item_popularity /= self.item_popularity.sum()

    # ---------------- USER ENCODING ----------------
    def encode_user(self, batch_item_lists):
        """
        batch_item_lists: List[List[int]]
        returns:
            gmf_u: [B, D]
            mlp_u: [B, D]
        """

        gmf_out = []
        mlp_out = []

        for item_list in batch_item_lists:

            if len(item_list) == 0:
                zeros = torch.zeros((1, self._emb_dim), device=self.device)
                gmf_out.append(zeros)
                mlp_out.append(zeros)
                continue

            items = torch.tensor(item_list, device=self.device)

            gmf_out.append(
                self.gmf_item_emb(items).mean(dim=0, keepdim=True)
            )
            mlp_out.append(
                self.mlp_item_emb(items).mean(dim=0, keepdim=True)
            )

        return torch.cat(gmf_out, dim=0), torch.cat(mlp_out, dim=0)

    # ---------------- SCORING ----------------
    def score(self, gmf_u, gmf_i, mlp_u, mlp_i):

        gmf_score = (gmf_u * gmf_i).sum(dim=1)

        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_score = self.mlp(mlp_in).squeeze(-1)

        return gmf_score + mlp_score

    # ---------------- HARD NEGATIVE SAMPLER ----------------
    def sample_hard_negative(self, history, positive_item, k=50):

        forbidden = set(history)
        forbidden.add(positive_item)

        mask = np.ones(self._num_items, dtype=bool)
        mask[list(forbidden)] = False

        allowed_items = np.where(mask)[0]

        if len(allowed_items) == 0:
            return np.random.randint(self._num_items)

        # -------------------------
        # hybrid sampling
        # -------------------------
        if np.random.rand() < 0.8:
            probs = self.item_popularity[allowed_items]
            probs = probs / probs.sum()

            sampled = np.random.choice(
                allowed_items,
                size=min(k, len(allowed_items)),
                replace=False,
                p=probs
            )
        else:
            sampled = np.random.choice(
                allowed_items,
                size=min(k, len(allowed_items)),
                replace=False
            )

        items = torch.tensor(sampled, device=self.device)

        with torch.no_grad():

            gmf_u, mlp_u = self.encode_user([history])

            gmf_i = self.gmf_item_emb(items)
            mlp_i = self.mlp_item_emb(items)

            scores = self.score(
                gmf_u.expand(len(items), -1),
                gmf_i,
                mlp_u.expand(len(items), -1),
                mlp_i
            )

        return items[torch.argmax(scores)].item()

    # ---------------- TRAIN ----------------
    def fit(self):

        print("_____TRAINING_____", flush=True)

        batch_size = 1024

        for epoch in range(self._epochs):

            print(f"_____STARTING_EPOCH_{epoch}_____", flush=True)

            np.random.shuffle(self.pos_interactions)

            epoch_loss = 0.0

            for start in range(0, len(self.pos_interactions), batch_size):

                batch = self.pos_interactions[start:start + batch_size]

                batch_histories = []
                pos_items = []
                neg_items = []

                for u, pos_i in batch:
                    history = [i for i in self.user_history[u] if i != pos_i]

                    if len(history) == 0:
                        continue

                    batch_histories.append(history)
                    pos_items.append(pos_i)

                    neg_items.append(
                        self.sample_hard_negative(history, pos_i)
                    )

                # Entire batch may have been skipped
                if len(pos_items) == 0:
                    continue

                pi = torch.tensor(pos_items, device=self.device)
                ni = torch.tensor(neg_items, device=self.device)

                gmf_u, mlp_u = self.encode_user(batch_histories)

                gmf_pi = self.gmf_item_emb(pi)
                gmf_ni = self.gmf_item_emb(ni)

                mlp_pi = self.mlp_item_emb(pi)
                mlp_ni = self.mlp_item_emb(ni)

                pos_score = self.score(gmf_u, gmf_pi, mlp_u, mlp_pi)
                neg_score = self.score(gmf_u, gmf_ni, mlp_u, mlp_ni)

                loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch} | loss = {epoch_loss:.4f}")

    # ---------------- PREDICT ----------------
    def predict(self, selected_items, filter_out_items, k):

        print("_____PREDICTING_____", flush=True)
        print("Selected items: ", flush=True)
        for item in selected_items:
            print(item, flush=True)

        selected_items = [
            self.item2idx[i] for i in selected_items if i in self.item2idx
        ]

        filter_out_items = set(
            self.item2idx[i] for i in filter_out_items if i in self.item2idx
        )

        candidates = np.setdiff1d(
            np.arange(self._num_items),
            np.union1d(selected_items, list(filter_out_items))
        )

        if len(candidates) == 0:
            return []

        with torch.no_grad():

            cand = torch.tensor(candidates, device=self.device)

            gmf_u, mlp_u = self.encode_user([selected_items])

            gmf_i = self.gmf_item_emb(cand)
            mlp_i = self.mlp_item_emb(cand)

            scores = self.score(
                gmf_u.expand(len(cand), -1),
                gmf_i,
                mlp_u.expand(len(cand), -1),
                mlp_i
            )

            scores = torch.sigmoid(scores)

            topk = torch.topk(
                scores,
                k=min(k, len(candidates))
            ).indices.cpu().numpy()

        return [self.idx2item[candidates[i]] for i in topk]

    # ---------------- META ----------------
    @classmethod
    def name(cls):
        return "NCF"

    @classmethod
    def parameters(cls):
        return [
            Parameter("emb_dim", ParameterType.INT, 16),
            Parameter("lr", ParameterType.FLOAT, 0.001),
            Parameter("epochs", ParameterType.INT, 5),
            Parameter("positive_threshold", ParameterType.FLOAT, 2.5),
        ]