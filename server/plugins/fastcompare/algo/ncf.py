import numpy as np
from abc import ABC

import torch
import torch.nn as nn

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class NCF(AlgorithmBase, ABC):

    def __init__(self, loader, positive_threshold, emb_dim=16, lr=1e-3, epochs=5, device=None, **kwargs):

        print("_____INITIALIZING_MODEL_____", flush=True)

        self._loader = loader
        self._ratings_df = loader.ratings_df

        self._threshold = positive_threshold
        self._emb_dim = emb_dim
        self._lr = lr
        self._epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- build matrix once ---
        self._rating_matrix = (
            self._ratings_df
            .pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._num_users, self._num_items = self._rating_matrix.shape

        self._build_model()
        self._prepare_data()

    # ---------------- MODEL ----------------
    def _build_model(self):

        print("_____BUILDING_MODEL_____", flush=True)

        self.user_emb = nn.Embedding(self._num_users, self._emb_dim).to(self.device)
        self.item_emb = nn.Embedding(self._num_items, self._emb_dim).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(self._emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

        self.loss_fn = nn.BCELoss()

        params = list(self.user_emb.parameters()) + list(self.item_emb.parameters()) + list(self.mlp.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self._lr)

    # ---------------- DATA PREP ----------------
    def _prepare_data(self):

        print("_____PREPATING_DATA_____", flush=True)

        X = (self._rating_matrix >= self._threshold).astype(np.int32)

        self.user_pos = {}
        self.train_users = []
        self.train_items = []
        self.train_labels = []

        all_items = np.arange(self._num_items)

        for u in range(self._num_users):

            pos_items = np.where(X[u] == 1)[0]

            if len(pos_items) == 0:
                continue

            self.user_pos[u] = pos_items

            for i in pos_items:

                # positive sample
                self.train_users.append(u)
                self.train_items.append(i)
                self.train_labels.append(1.0)

                # fast negative sampling (vectorized)
                neg = np.random.choice(all_items)
                while neg in pos_items:
                    neg = np.random.choice(all_items)

                self.train_users.append(u)
                self.train_items.append(neg)
                self.train_labels.append(0.0)

        # convert to tensors once
        self.train_users = torch.tensor(self.train_users, dtype=torch.long, device=self.device)
        self.train_items = torch.tensor(self.train_items, dtype=torch.long, device=self.device)
        self.train_labels = torch.tensor(self.train_labels, dtype=torch.float32, device=self.device)

    # ---------------- TRAINING ----------------
    def fit(self):

        batch_size = 8192

        n = len(self.train_users)

        for epoch in range(self._epochs):

            print(f"_____STARTING_EPOCH_{epoch}_____", flush=True)

            self.user_emb.train()
            self.item_emb.train()
            self.mlp.train()

            epoch_loss = 0.0
            total_samples = 0          # ← ADD

            for start in range(0, n, batch_size):

                end = min(start + batch_size, n)

                users = self.train_users[start:end]
                items = self.train_items[start:end]
                labels = self.train_labels[start:end]

                pred = self.forward(users, items)

                loss = self.loss_fn(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(users)
                total_samples += len(users)

                # progress every ~100 batches
                if start % (batch_size * 100) == 0:
                    print(
                        f"{start}/{n}",
                        flush=True
                    )

            avg_loss = epoch_loss / total_samples

            print(
                f"Epoch {epoch} | avg_loss = {avg_loss:.4f}",
                flush=True
            )

        print("Training finished", flush=True)

    # ---------------- FORWARD ----------------
    def forward(self, users, items):

        u = self.user_emb(users)
        i = self.item_emb(items)

        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)

        return torch.sigmoid(x).squeeze(-1)

    # ---------------- PREDICTION ----------------
    def predict(self, selected_items, filter_out_items, k):

        print("_____CALLING_PREDICT_____", flush=True)

        selected_items = list(selected_items)
        filter_out_items = set(filter_out_items)

        candidates = np.setdiff1d(
            np.arange(self._num_items),
            np.union1d(selected_items, list(filter_out_items))
        )

        if len(selected_items) == 0:
            return np.random.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()

        with torch.no_grad():

            user_vec = self.item_emb(torch.tensor(selected_items, device=self.device)).mean(dim=0)
            item_vecs = self.item_emb(torch.tensor(candidates, device=self.device))

            # vectorized scoring (FAST)
            x = torch.cat([
                user_vec.unsqueeze(0).expand(len(candidates), -1),
                item_vecs
            ], dim=1)

            scores = self.mlp(x).squeeze(-1)
            scores = torch.sigmoid(scores)

            topk = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().numpy()

        return candidates[topk].tolist()

    # ---------------- METADATA ----------------
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