import numpy as np

from abc import ABC
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

import torch
import torch.nn as nn


class NCF(AlgorithmBase, ABC):

    def __init__(self, loader, positive_threshold, emb_dim=32, lr=1e-3, epochs=10, **kwargs):

        self._loader = loader
        self._ratings_df = loader.ratings_df

        self._threshold = positive_threshold
        self._emb_dim = emb_dim
        self._lr = lr
        self._epochs = epochs

        # EASE-style item universe (optional, mostly for consistency/debugging)
        self._all_items = self._ratings_df.item.unique()

        # Build rating matrix (EASE-style preprocessing)
        self._rating_matrix = (
            self._ratings_df
            .pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._num_users, self._num_items = self._rating_matrix.shape

        self._build_model()

    # ---------------- MODEL ----------------
    def _build_model(self):

        self.user_emb = nn.Embedding(self._num_users, self._emb_dim)
        self.item_emb = nn.Embedding(self._num_items, self._emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self._emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()

        params = (
            list(self.user_emb.parameters()) +
            list(self.item_emb.parameters()) +
            list(self.mlp.parameters())
        )

        self.optimizer = torch.optim.Adam(params, lr=self._lr)
        self.loss_fn = nn.BCELoss()

    # ---------------- TRAINING ----------------
    def fit(self):

        # EASE-style binarization
        X = (self._rating_matrix >= self._threshold).astype(np.float32)

        user_pos = {
            u: set(np.where(X[u] == 1)[0])
            for u in range(self._num_users)
        }

        for epoch in range(self._epochs):

            print(f"Epoch {epoch} started")

            total_loss = 0.0

            for u in range(self._num_users):

                if u % 100 == 0:
                    print(f"user {u}/{self._num_users}")

                pos_items = user_pos[u]

                if len(pos_items) == 0:
                    continue

                for i in pos_items:

                    # positive sample
                    total_loss += self._train_step(u, i, 1.0)

                    # negative sample
                    j = self._sample_negative(pos_items)
                    total_loss += self._train_step(u, j, 0.0)

            print(f"Epoch {epoch}: {total_loss:.4f}")

    def _train_step(self, user, item, label):

        u = torch.tensor([user], dtype=torch.long)
        i = torch.tensor([item], dtype=torch.long)
        y = torch.tensor([label], dtype=torch.float32)

        pred = self.forward(u, i)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _sample_negative(self, pos_items):

        while True:
            j = np.random.randint(self._num_items)
            if j not in pos_items:
                return j

    # ---------------- FORWARD ----------------
    def forward(self, user, item):

        u = self.user_emb(user)
        i = self.item_emb(item)

        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)

        return self.sigmoid(x).view(-1)

    # ---------------- PREDICTION ----------------
    def predict(self, selected_items, filter_out_items, k):

        selected_items = list(selected_items)
        filter_out_items = set(filter_out_items)

        # candidate filtering (EASE-consistent)
        candidates = np.setdiff1d(
            np.arange(self._num_items),
            np.union1d(selected_items, list(filter_out_items))
        )

        # cold-start fallback (EASE behavior)
        if len(selected_items) == 0:
            if len(candidates) == 0:
                return []

            return np.random.choice(
                candidates,
                size=min(k, len(candidates)),
                replace=False
            ).tolist()

        with torch.no_grad():

            item_vecs = self.item_emb(torch.tensor(candidates, dtype=torch.long))
            selected_vecs = self.item_emb(torch.tensor(selected_items, dtype=torch.long))

            user_vec = selected_vecs.mean(dim=0)

            scores = []

            for idx, item in enumerate(candidates):

                x = torch.cat([
                    user_vec.unsqueeze(0),
                    item_vecs[idx].unsqueeze(0)
                ], dim=-1)

                score = self.sigmoid(self.mlp(x)).item()
                scores.append((score, int(item)))

        scores.sort(reverse=True, key=lambda x: x[0])

        return [item for _, item in scores[:k]]

    # ---------------- METADATA ----------------
    @classmethod
    def name(cls):
        return "NCF"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "emb_dim",
                ParameterType.INT,
                32,
                help="Dimensionality of embeddings",
            ),
            Parameter(
                "lr",
                ParameterType.FLOAT,
                0.001,
                help="Learning rate for optimizer",
            ),
            Parameter(
                "epochs",
                ParameterType.INT,
                10,
                help="Number of training epochs",
            ),
            Parameter(
                "positive_threshold",
                ParameterType.FLOAT,
                2.5,
                help="Threshold for converting ratings into implicit feedback",
            ),
        ]
