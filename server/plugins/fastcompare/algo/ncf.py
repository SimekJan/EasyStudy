import numpy as np
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # ---------------- SAFE INDEXING ----------------
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

        self.gmf_user_emb = nn.Embedding(self._num_users, self._emb_dim)
        self.gmf_item_emb = nn.Embedding(self._num_items, self._emb_dim)

        self.mlp_user_emb = nn.Embedding(self._num_users, self._emb_dim)
        self.mlp_item_emb = nn.Embedding(self._num_items, self._emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self._emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # NOTE: no self.to(self.device)

        params = (
            list(self.gmf_user_emb.parameters()) +
            list(self.gmf_item_emb.parameters()) +
            list(self.mlp_user_emb.parameters()) +
            list(self.mlp_item_emb.parameters()) +
            list(self.mlp.parameters())
        )

        self.optimizer = torch.optim.Adam(params, lr=self._lr)

    # ---------------- DATA PREP ----------------
    def _prepare_data(self):

        print("_____PREPARING_DATA_____", flush=True)

        self.train_users = []
        self.train_pos = []
        self.train_neg = []
        self.user_pos = {}

        df = self._ratings_df[["user", "item", "rating"]]

        for u_raw, i_raw, r in df.itertuples(index=False):

            if r < self._threshold:
                continue

            # SAFE mapping
            if u_raw not in self.user2idx or i_raw not in self.item2idx:
                continue

            u = self.user2idx[u_raw]
            i = self.item2idx[i_raw]

            self.train_users.append(u)
            self.train_pos.append(i)
            self.user_pos.setdefault(u, set()).add(i)

        # ---------------- NEGATIVE SAMPLING ----------------
        all_items = np.arange(self._num_items)

        for u, pos_i in zip(self.train_users, self.train_pos):

            neg = np.random.randint(self._num_items)
            # safe fallback (bounded loop)
            tries = 0
            while neg in self.user_pos[u] and tries < 20:
                neg = np.random.randint(self._num_items)
                tries += 1

            self.train_neg.append(neg)

        # tensors
        self.train_users = torch.tensor(self.train_users, dtype=torch.long, device=self.device)
        self.train_pos = torch.tensor(self.train_pos, dtype=torch.long, device=self.device)
        self.train_neg = torch.tensor(self.train_neg, dtype=torch.long, device=self.device)

    # ---------------- FORWARD ----------------
    def forward(self, users, items):

        gmf_u = self.gmf_user_emb(users)
        gmf_i = self.gmf_item_emb(items)
        gmf = (gmf_u * gmf_i).sum(dim=1, keepdim=True)

        mlp_u = self.mlp_user_emb(users)
        mlp_i = self.mlp_item_emb(items)

        mlp = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        return gmf + mlp

    # ---------------- TRAIN ----------------
    def fit(self):

        batch_size = 8192
        n = len(self.train_users)

        for epoch in range(self._epochs):

            print(f"_____STARTING_EPOCH_{epoch}_____", flush=True)

            epoch_loss = 0.0

            for start in range(0, n, batch_size):

                end = min(start + batch_size, n)

                u = self.train_users[start:end]
                p = self.train_pos[start:end]
                n_i = self.train_neg[start:end]

                pos_scores = self.forward(u, p)
                neg_scores = self.forward(u, n_i)

                loss = -F.logsigmoid(pos_scores - neg_scores).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch} | loss = {epoch_loss:.4f}", flush=True)

        print("Training finished", flush=True)

    # ---------------- PREDICT ----------------
    def predict(self, selected_items, filter_out_items, k):

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

        if len(selected_items) == 0 or len(candidates) == 0:
            return np.random.choice(
                candidates if len(candidates) > 0 else np.arange(self._num_items),
                size=min(k, len(candidates)) if len(candidates) > 0 else k,
                replace=False
            ).tolist()

        with torch.no_grad():

            cand = torch.tensor(candidates, device=self.device)
            selected = torch.tensor(selected_items, device=self.device)

            # GMF
            user_gmf = self.gmf_item_emb(selected).mean(dim=0)
            item_gmf = self.gmf_item_emb(cand)
            gmf = (user_gmf * item_gmf).sum(dim=1)

            # MLP
            user_mlp = self.mlp_item_emb(selected).mean(dim=0)
            item_mlp = self.mlp_item_emb(cand)

            mlp_in = torch.cat([
                user_mlp.unsqueeze(0).expand(len(cand), -1),
                item_mlp
            ], dim=1)

            mlp = self.mlp(mlp_in).squeeze(-1)

            scores = gmf + mlp

            topk = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().numpy()

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