"""
Hierarchical Thompson Sampling:
- livello 1: scelta cluster con Beta-Bernoulli TS
- livello 2: scelta canzone nel cluster con Linear TS (posterior Bayesiano tipo ridge)

Versione adattata per track_id STRING (nel tuo CSV track_id è alfanumerico).
Deriva dal tuo thomp_samp.py ma con:
- id non castati a int
- stima di probabilità "user-like" combinando cluster+song
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from config import PoolConfig


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class FullHierarchicalTS:
    def __init__(
        self,
        df_songs: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "track_id",
        cluster_col: str = "cluster",
        pool_cfg: PoolConfig = PoolConfig(),
        seed: int = 42,
        lambda_prior: float = 1.0,
        sigma2: float = 1.0
    ):
        self.rng = np.random.default_rng(seed)

        self.df = df_songs.copy()
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.cluster_col = cluster_col
        self.pool_cfg = pool_cfg

        # ensure float features
        self.df[self.feature_cols] = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

        self.ids = self.df[self.id_col].astype(str).values
        self.id_to_row = dict(zip(self.ids, self.df.index.values))

        self.clusters = sorted(self.df[self.cluster_col].unique().tolist())
        self.K = len(self.clusters)
        self.cluster_to_idx = {c: i for i, c in enumerate(self.clusters)}
        self.idx_to_cluster = {i: c for c, i in self.cluster_to_idx.items()}

        # --- TS cluster: Beta priors ---
        self.alpha = np.ones(self.K, dtype=float)
        self.beta = np.ones(self.K, dtype=float)

        # --- Linear TS: Bayesian ridge posterior ---
        self.d = len(self.feature_cols)
        self.lambda_prior = float(lambda_prior)
        self.sigma2 = float(sigma2)
        self.A = self.lambda_prior * np.eye(self.d, dtype=float)
        self.b = np.zeros(self.d, dtype=float)

        self.seen_ids: set[str] = set()
        self.pools: Dict[int, np.ndarray] = {}
        self._updates = 0

        # precompute cluster geometry
        self._cluster_members: Dict[int, np.ndarray] = {}
        self._cluster_dist_to_centroid: Dict[int, np.ndarray] = {}

        self._precompute_cluster_geometry()
        self._build_candidate_pools()

    def _precompute_cluster_geometry(self) -> None:
        X = self.df[self.feature_cols].to_numpy(dtype=float)

        for c in self.clusters:
            k = self.cluster_to_idx[c]
            member_idx = np.where(self.df[self.cluster_col].values == c)[0]
            self._cluster_members[k] = member_idx

            if len(member_idx) == 0:
                self._cluster_dist_to_centroid[k] = np.array([])
                continue

            Xk = X[member_idx]
            centroid = Xk.mean(axis=0)
            dist = np.linalg.norm(Xk - centroid, axis=1)
            self._cluster_dist_to_centroid[k] = dist

    def _build_candidate_pools(self) -> None:
        """
        Pool adattivo:
        - se il cluster è piccolo, ridimensiona m_center/m_diverse/m_random
        - evita duplicati per costruzione
        Nota: l'esclusione dei "seen" avviene in select_song().
        """
        ids = self.df[self.id_col].astype(str).values
        self.pools = {}

        for k in range(self.K):
            member_idx = self._cluster_members[k]
            n = len(member_idx)

            if n == 0:
                self.pools[k] = np.array([], dtype=ids.dtype)
                continue

            dist = self._cluster_dist_to_centroid[k]
            order = np.argsort(dist)  # vicino -> lontano

            m_center = self.pool_cfg.m_center
            m_diverse = self.pool_cfg.m_diverse
            m_random = self.pool_cfg.m_random

            total_desired = m_center + m_diverse + m_random
            if n < total_desired:
                scale = n / max(total_desired, 1)
                m_center = int(np.floor(m_center * scale))
                m_diverse = int(np.floor(m_diverse * scale))
                m_random = n - (m_center + m_diverse)

                if n == 1:
                    m_center, m_diverse, m_random = 1, 0, 0
                elif n == 2:
                    m_center, m_diverse, m_random = 1, 1, 0
                elif n >= 3 and (m_center + m_diverse) == 0:
                    m_center, m_diverse = 1, 1
                    m_random = max(0, n - 2)

            near_idx = member_idx[order[: min(m_center, n)]]
            far_idx = member_idx[order[::-1][: min(m_diverse, n)]]

            taken = np.unique(np.concatenate([near_idx, far_idx]))
            remaining = np.setdiff1d(member_idx, taken, assume_unique=False)

            m_rand = min(m_random, len(remaining))
            rand_idx = self.rng.choice(remaining, size=m_rand, replace=False) if m_rand > 0 else np.array([], dtype=int)

            pool_idx = np.unique(np.concatenate([near_idx, far_idx, rand_idx]))
            self.pools[k] = ids[pool_idx]

    def initialize_from_cold_start(self, user_history: pd.DataFrame, label_col: str = "label") -> None:
        for _, row in user_history.iterrows():
            tid = str(row[self.id_col])
            if tid not in self.id_to_row:
                continue
            self._update_internal(tid, int(row[label_col]))

    def select_cluster(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def _posterior_mean_theta(self) -> np.ndarray:
        A_inv = np.linalg.inv(self.A)
        return A_inv @ self.b

    def _sample_theta(self) -> np.ndarray:
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        Sigma = self.sigma2 * A_inv
        return self.rng.multivariate_normal(mu, Sigma)

    def select_song(self, cluster_idx: int) -> Optional[str]:
        """
        - con probabilità epsilon_out_of_pool: random nel cluster (non visto)
        - altrimenti: Linear TS nel pool
        - fallback robusto: se pool vuoto/esaurito -> Linear TS su tutto il cluster
        """
        member_rows = self._cluster_members.get(cluster_idx, np.array([], dtype=int))
        if len(member_rows) == 0:
            return None

        member_ids = self.df.iloc[member_rows][self.id_col].astype(str).values
        unseen_member_ids = [t for t in member_ids if t not in self.seen_ids]
        if len(unseen_member_ids) == 0:
            return None

        if self.rng.random() < float(self.pool_cfg.epsilon_out_of_pool):
            return str(self.rng.choice(unseen_member_ids))

        pool_ids = self.pools.get(cluster_idx, np.array([], dtype=member_ids.dtype)).astype(str)
        unseen_pool_ids = [t for t in pool_ids if t not in self.seen_ids]

        candidate_ids = unseen_pool_ids if len(unseen_pool_ids) > 0 else unseen_member_ids

        theta_tilde = self._sample_theta()
        rows = [self.id_to_row[tid] for tid in candidate_ids]
        Xc = self.df.loc[rows, self.feature_cols].to_numpy(dtype=float)
        scores = Xc @ theta_tilde

        return str(candidate_ids[int(np.argmax(scores))])

    def update(self, track_id: str, reward: int) -> None:
        self._update_internal(str(track_id), int(reward))
        self._updates += 1
        if self.pool_cfg.refresh_every > 0 and (self._updates % self.pool_cfg.refresh_every == 0):
            self._build_candidate_pools()

    def _update_internal(self, track_id: str, reward: int) -> None:
        if reward not in (0, 1):
            raise ValueError("reward deve essere 0 o 1")

        self.seen_ids.add(track_id)

        row_idx = self.id_to_row[track_id]
        c = int(self.df.loc[row_idx, self.cluster_col])
        k = self.cluster_to_idx[c]

        # Beta update (cluster)
        if reward == 1:
            self.alpha[k] += 1.0
        else:
            self.beta[k] += 1.0

        # Linear TS update (song features)
        x = self.df.loc[row_idx, self.feature_cols].to_numpy(dtype=float)
        self.A += np.outer(x, x) / self.sigma2
        self.b += (x * reward) / self.sigma2

    def recommend_one(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Ritorna: (track_id, cluster_idx). Se un cluster è esaurito, prova altri cluster.
        """
        k = self.select_cluster()
        tid = self.select_song(k)
        if tid is not None:
            return tid, k

        samples = self.rng.beta(self.alpha, self.beta)
        for kk in np.argsort(samples)[::-1]:
            tid = self.select_song(int(kk))
            if tid is not None:
                return tid, int(kk)

        return None, None

    def predict_like_probability(self, track_id: str, cluster_idx: Optional[int] = None) -> float:
        """
        Stima una "probabilità che ti piaccia" usando:
        - p_cluster = E[theta_cluster] = alpha/(alpha+beta)
        - p_song = sigmoid( mu^T x ) con mu = media posterior del Linear TS
        Combino con prodotto (conservativo).
        """
        tid = str(track_id)
        if tid not in self.id_to_row:
            return 0.0
        row_idx = self.id_to_row[tid]

        if cluster_idx is None:
            c = int(self.df.loc[row_idx, self.cluster_col])
            cluster_idx = self.cluster_to_idx[c]

        p_cluster = float(self.alpha[cluster_idx] / (self.alpha[cluster_idx] + self.beta[cluster_idx]))

        mu = self._posterior_mean_theta()
        x = self.df.loc[row_idx, self.feature_cols].to_numpy(dtype=float)
        p_song = float(sigmoid(float(x @ mu)))

        # prodotto: alto solo se entrambi sono alti
        p = p_cluster * p_song
        # clamp numerico
        return float(max(0.0, min(1.0, p)))
