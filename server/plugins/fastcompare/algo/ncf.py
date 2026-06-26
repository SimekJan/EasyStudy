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

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(self._num_users, self._emb_dim)
        self.gmf_item_emb = nn.Embedding(self._num_items, self._emb_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(self._num_users, self._emb_dim)
        self.mlp_item_emb = nn.Embedding(self._num_items, self._emb_dim)

        # MLP tower
        self.mlp = nn.Sequential(
            nn.Linear(self._emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        params = (
            list(self.gmf_user_emb.parameters()) +
            list(self.gmf_item_emb.parameters()) +
            list(self.mlp_user_emb.parameters()) +
            list(self.mlp_item_emb.parameters()) +
            list(self.mlp.parameters())
        )

        self.optimizer = torch.optim.Adam(params, lr=self._lr)

    # ---------------- DATA ----------------
    def _prepare_data(self):

        print("_____PREPARING_DATA_____", flush=True)

        self.pos_interactions = []

        df = self._ratings_df[["user", "item", "rating"]]

        for u_raw, i_raw, r in df.itertuples(index=False):

            if u_raw not in self.user2idx or i_raw not in self.item2idx:
                continue

            if r >= self._threshold:
                u = self.user2idx[u_raw]
                i = self.item2idx[i_raw]
                self.pos_interactions.append((u, i))

    # ---------------- SCORING (CRITICAL FIX) ----------------
    def score(self, gmf_u, gmf_i, mlp_u, mlp_i):

        gmf_score = (gmf_u * gmf_i).sum(dim=1)

        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_score = self.mlp(mlp_in).squeeze(-1)

        return gmf_score + mlp_score

    # ---------------- HARD NEGATIVE SAMPLER ----------------
    def sample_hard_negative(self, u, k=50):

        u_tensor = torch.tensor([u], device=self.device)

        items = torch.randint(0, self._num_items, (k,), device=self.device)

        with torch.no_grad():

            gmf_u = self.gmf_user_emb(u_tensor)
            mlp_u = self.mlp_user_emb(u_tensor)

            gmf_i = self.gmf_item_emb(items)
            mlp_i = self.mlp_item_emb(items)

            gmf_score = (gmf_u * gmf_i).sum(dim=1)

            mlp_in = torch.cat([
                mlp_u.expand(k, -1),
                mlp_i
            ], dim=1)

            mlp_score = self.mlp(mlp_in).squeeze(-1)

            scores = gmf_score + mlp_score

        return items[torch.argmax(scores)].item()

    # ---------------- TRAIN ----------------
    def fit(self):

        batch_size = 1024

        for epoch in range(self._epochs):

            print(f"_____STARTING_EPOCH_{epoch}_____", flush=True)

            np.random.shuffle(self.pos_interactions)

            epoch_loss = 0.0

            for start in range(0, len(self.pos_interactions), batch_size):

                batch = self.pos_interactions[start:start + batch_size]

                users = []
                pos_items = []
                neg_items = []

                for (u, pos_i) in batch:

                    users.append(u)
                    pos_items.append(pos_i)

                    neg_i = self.sample_hard_negative(u)
                    neg_items.append(neg_i)

                u = torch.tensor(users, device=self.device)
                pi = torch.tensor(pos_items, device=self.device)
                ni = torch.tensor(neg_items, device=self.device)

                # embeddings
                gmf_u = self.gmf_user_emb(u)

                gmf_pi = self.gmf_item_emb(pi)
                gmf_ni = self.gmf_item_emb(ni)

                mlp_u = self.mlp_user_emb(u)

                mlp_pi = self.mlp_item_emb(pi)
                mlp_ni = self.mlp_item_emb(ni)

                # 🔥 unified scoring (FIXED)
                pos_score = self.score(gmf_u, gmf_pi, mlp_u, mlp_pi)
                neg_score = self.score(gmf_u, gmf_ni, mlp_u, mlp_ni)

                # BPR loss
                loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch} | loss = {epoch_loss:.4f}")

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

        if len(candidates) == 0:
            return []

        with torch.no_grad():

            cand = torch.tensor(candidates, device=self.device)
            selected = torch.tensor(selected_items, device=self.device)

            # 🔥 consistent representation with training (important fix)
            gmf_u = self.gmf_item_emb(selected).mean(dim=0, keepdim=True)
            mlp_u = self.mlp_item_emb(selected).mean(dim=0, keepdim=True)

            gmf_i = self.gmf_item_emb(cand)
            mlp_i = self.mlp_item_emb(cand)

            scores = self.score(
                gmf_u.expand(len(cand), -1),
                gmf_i,
                mlp_u.expand(len(cand), -1),
                mlp_i
            )

            scores = torch.sigmoid(scores)

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