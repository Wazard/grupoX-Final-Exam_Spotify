import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class PoolConfig:
    m_center: int = 200
    m_diverse: int = 200
    m_random: int = 100
    refresh_every: int = 25
    epsilon_out_of_pool: float = 0.15  # % volte scegli random fuori-pool (ma nel cluster)


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

        self.df[self.feature_cols] = self.df[self.feature_cols].astype(float).fillna(0.0)

        self.ids = self.df[self.id_col].values
        self.id_to_row = dict(zip(self.ids, self.df.index.values))

        self.clusters = sorted(self.df[self.cluster_col].unique().tolist())
        self.K = len(self.clusters)
        self.cluster_to_idx = {c: i for i, c in enumerate(self.clusters)}

        # --- TS cluster: Beta priors ---
        self.alpha = np.ones(self.K, dtype=float)
        self.beta = np.ones(self.K, dtype=float)

        # --- Linear TS globale: Bayesian ridge posterior ---
        self.d = len(self.feature_cols)
        self.lambda_prior = float(lambda_prior)
        self.sigma2 = float(sigma2)
        self.A = self.lambda_prior * np.eye(self.d, dtype=float)
        self.b = np.zeros(self.d, dtype=float)

        self.seen_ids = set()
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
        - se il cluster è piccolo, ridimensiona m_center/m_diverse/m_random automaticamente
        - evita errori e duplicati
        """
        ids = self.df[self.id_col].values
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

            # ---- (1) RIDIMENSIONAMENTO AUTOMATICO se cluster piccolo ----
            total_desired = m_center + m_diverse + m_random
            if n < total_desired:
                scale = n / max(total_desired, 1)
                m_center = int(np.floor(m_center * scale))
                m_diverse = int(np.floor(m_diverse * scale))
                m_random = n - (m_center + m_diverse)

                # protezioni per cluster molto piccoli
                if n == 1:
                    m_center, m_diverse, m_random = 1, 0, 0
                elif n == 2:
                    m_center, m_diverse, m_random = 1, 1, 0
                elif n >= 3 and (m_center + m_diverse) == 0:
                    m_center, m_diverse = 1, 1
                    m_random = max(0, n - 2)

            near_idx = member_idx[order[: min(m_center, n)]]
            far_idx = member_idx[order[::-1][: min(m_diverse, n)]]

            # ---- (2) NIENTE DUPLICATI: random solo dai rimanenti ----
            taken = np.unique(np.concatenate([near_idx, far_idx]))
            remaining = np.setdiff1d(member_idx, taken, assume_unique=False)

            m_rand = min(m_random, len(remaining))
            rand_idx = self.rng.choice(remaining, size=m_rand, replace=False) if m_rand > 0 else np.array([], dtype=int)

            pool_idx = np.unique(np.concatenate([near_idx, far_idx, rand_idx]))
            self.pools[k] = ids[pool_idx]

    def initialize_from_cold_start(self, user_history: pd.DataFrame, label_col: str = "label") -> None:
        for _, row in user_history.iterrows():
            tid = row[self.id_col]
            if tid not in self.id_to_row:
                continue
            self._update_internal(int(tid), int(row[label_col]))

    def select_cluster(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def _sample_theta(self) -> np.ndarray:
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        Sigma = self.sigma2 * A_inv
        return self.rng.multivariate_normal(mu, Sigma)

    def select_song(self, cluster_idx: int) -> Optional[int]:
        """
        - con probabilità epsilon_out_of_pool: random nel cluster (non visto)
        - altrimenti: Linear TS nel pool
        - fallback robusto: se pool vuoto/esaurito -> Linear TS su tutto il cluster
        """
        member_rows = self._cluster_members.get(cluster_idx, np.array([], dtype=int))
        if len(member_rows) == 0:
            return None

        member_ids = self.df.iloc[member_rows][self.id_col].values
        unseen_member_ids = [int(t) for t in member_ids if int(t) not in self.seen_ids]
        if len(unseen_member_ids) == 0:
            return None

        # out-of-pool safeguard
        if self.rng.random() < float(self.pool_cfg.epsilon_out_of_pool):
            return int(self.rng.choice(unseen_member_ids))

        pool_ids = self.pools.get(cluster_idx, np.array([], dtype=member_ids.dtype))
        unseen_pool_ids = [int(t) for t in pool_ids if int(t) not in self.seen_ids]

        # ---- (3) FALLBACK: se pool esaurito, usa tutto il cluster ----
        candidate_ids = unseen_pool_ids if len(unseen_pool_ids) > 0 else unseen_member_ids

        theta_tilde = self._sample_theta()

        rows = [self.id_to_row[tid] for tid in candidate_ids]
        Xc = self.df.loc[rows, self.feature_cols].to_numpy(dtype=float)
        scores = Xc @ theta_tilde

        return int(candidate_ids[int(np.argmax(scores))])

    def update(self, track_id: int, reward: int) -> None:
        self._update_internal(int(track_id), int(reward))
        self._updates += 1
        if self.pool_cfg.refresh_every > 0 and (self._updates % self.pool_cfg.refresh_every == 0):
            self._build_candidate_pools()

    def _update_internal(self, track_id: int, reward: int) -> None:
        if reward not in (0, 1):
            raise ValueError("reward deve essere 0 o 1")

        self.seen_ids.add(track_id)

        row_idx = self.id_to_row[track_id]
        c = self.df.loc[row_idx, self.cluster_col]
        k = self.cluster_to_idx[c]

        # Beta update
        if reward == 1:
            self.alpha[k] += 1.0
        else:
            self.beta[k] += 1.0

        # Linear TS update
        x = self.df.loc[row_idx, self.feature_cols].to_numpy(dtype=float)
        self.A += np.outer(x, x) / self.sigma2
        self.b += (x * reward) / self.sigma2

    def recommend_one(self) -> Tuple[Optional[int], Optional[int]]:
        k = self.select_cluster()
        tid = self.select_song(k)
        if tid is not None:
            return tid, k

        # fallback: prova cluster alternativi
        samples = self.rng.beta(self.alpha, self.beta)
        for kk in np.argsort(samples)[::-1]:
            tid = self.select_song(int(kk))
            if tid is not None:
                return tid, int(kk)

        return None, None
