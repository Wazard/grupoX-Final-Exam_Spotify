"""
benchmark.py

Benchmark per sistema di raccomandazione musicale.

Algoritmi confrontati:
1) random               - pesca random tra le canzoni non viste
2) popularity           - pesca la più popolare tra le non viste (globale)
3) ts_hier              - il tuo Hierarchical Thompson Sampling (cluster TS + linear TS sulle canzoni)
4) cluster_ts_hybrid    - Thompson Sampling SOLO sui cluster (Beta-Bernoulli),
                          poi dentro il cluster sceglie la canzone migliore usando:
                          popolarità + preferenza imparata su track_genre + preferenza imparata su artista.

Metriche salvate:
- cumulative_like_rate_mean.png
- rolling_like_rate_mean.png
- expected_like_rate_mean.png (usa p_true del simulatore, non il reward campionato)

Uso:
  python benchmark.py --csv tracks_with_clusters.csv --T 2000 --seeds 20 --out benchmark_results

Note:
- Non ripropone mai la stessa canzone.
- Il benchmark assume che esistano i moduli del tuo progetto:
  data_utils.py, cold_start_clustered.py, hierarchical_ts.py, sim_user.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_songs_csv, ensure_feature_cols
from cold_start_clustered import build_cold_start, ask_labels
from hierarchical_ts import FullHierarchicalTS
from sim_user import UserSimulator, UserSimConfig
from sim_user_genre_only_v2 import GenreOnlyUserSimulator, GenreOnlyUserConfig


# -----------------------
# Util
# -----------------------

def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, int(w))
    out = np.full_like(x, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        out[i] = (cumsum[i + 1] - cumsum[j0]) / (i - j0 + 1)
    return out


def _pad_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    if len(arr) >= T:
        return arr[:T]
    out = np.full(T, np.nan, dtype=float)
    out[: len(arr)] = arr
    return out


# -----------------------
# Runner: Random
# -----------------------

def run_random(df: pd.DataFrame, sim, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unseen = df.copy()
    rewards, p_trues = [], []

    for _t in range(T):
        if unseen.empty:
            break
        idx = rng.integers(0, len(unseen))
        row = unseen.iloc[idx]
        reward, p_true = sim.rate(row)
        rewards.append(float(reward))
        p_trues.append(float(p_true))
        unseen = unseen[unseen["track_id"] != row["track_id"]]

    return np.array(rewards, dtype=float), np.array(p_trues, dtype=float)


# -----------------------
# Runner: Popularity globale
# -----------------------

def run_popularity(df: pd.DataFrame, sim, T: int) -> Tuple[np.ndarray, np.ndarray]:
    unseen = df.copy()
    if "popularity" not in unseen.columns:
        unseen["popularity"] = 0
    unseen = unseen.sort_values("popularity", ascending=False)

    rewards, p_trues = [], []
    for _t in range(T):
        if unseen.empty:
            break
        row = unseen.iloc[0]
        reward, p_true = sim.rate(row)
        rewards.append(float(reward))
        p_trues.append(float(p_true))
        unseen = unseen.iloc[1:]

    return np.array(rewards, dtype=float), np.array(p_trues, dtype=float)


# -----------------------
# Runner: TS gerarchico (tuo)
# -----------------------

def run_ts_hier(df: pd.DataFrame, sim, feature_cols: List[str], T: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    # Cold start automatico col simulatore
    cold = build_cold_start(df, n_per_cluster=2)
    labeled = ask_labels(cold, df=df, auto_rater=sim)

    user_hist = pd.DataFrame(labeled)
    ts = FullHierarchicalTS(df_songs=df, feature_cols=feature_cols, seed=seed)
    ts.initialize_from_cold_start(user_history=user_hist, label_col="label")

    rewards, p_trues = [], []
    for _t in range(T):
        track_id, _cluster_idx = ts.recommend_one()
        if track_id is None:
            break
        row = df[df["track_id"].astype(str) == str(track_id)].iloc[0]
        reward, p_true = sim.rate(row)
        ts.update(track_id=str(track_id), reward=int(reward))
        rewards.append(float(reward))
        p_trues.append(float(p_true))

    return np.array(rewards, dtype=float), np.array(p_trues, dtype=float)


# -----------------------
# Runner: Cluster-TS + Hybrid in-cluster scoring
# -----------------------

@dataclass
class HybridConfig:
    w_pop: float = 0.55
    w_genre: float = 0.30
    w_artist: float = 0.15


def run_cluster_ts_hybrid(df: pd.DataFrame, sim, T: int, seed: int, cfg: Optional[HybridConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    - Thompson Sampling sui cluster (Beta-Bernoulli)
    - dentro cluster: rank deterministico con score = w_pop*pop_norm + w_genre*E[like|genre] + w_artist*E[like|artist]
    - aggiorna:
        cluster beta, genre beta, artist beta
    """
    if cfg is None:
        cfg = HybridConfig()

    rng = np.random.default_rng(seed)

    work = df.copy()
    if "cluster" not in work.columns:
        raise ValueError("Serve la colonna 'cluster' per cluster_ts_hybrid.")
    if "popularity" not in work.columns:
        work["popularity"] = 0

    work["track_id"] = work["track_id"].astype(str)
    work["track_genre"] = work.get("track_genre", "").astype(str)

    clusters = sorted(work["cluster"].dropna().unique().tolist())
    clusters = [int(c) for c in clusters]

    # Beta params per cluster
    a_c = {c: 1.0 for c in clusters}
    b_c = {c: 1.0 for c in clusters}

    # Beta params per genre/artist (inizio non-informativo)
    a_g: Dict[str, float] = {}
    b_g: Dict[str, float] = {}
    a_a: Dict[str, float] = {}
    b_a: Dict[str, float] = {}

    # precompute pop norm per cluster
    max_pop_by_cluster = work.groupby("cluster")["popularity"].max().to_dict()
    max_pop_global = float(work["popularity"].max() or 1.0)

    seen_ids = set()

    def mean_beta(a: float, b: float) -> float:
        return a / (a + b)

    def get_genre_mean(genre: str) -> float:
        genre = genre.strip().lower()
        if genre == "":
            return 0.5
        if genre not in a_g:
            a_g[genre], b_g[genre] = 1.0, 1.0
        return mean_beta(a_g[genre], b_g[genre])

    def get_artist_mean(artist: str) -> float:
        artist = artist.strip().lower()
        if artist == "":
            return 0.5
        if artist not in a_a:
            a_a[artist], b_a[artist] = 1.0, 1.0
        return mean_beta(a_a[artist], b_a[artist])

    rewards, p_trues = [], []

    for _t in range(T):
        # sample cluster thetas, but must have unseen items
        sampled = []
        for c in clusters:
            # if cluster has no unseen songs -> mark unusable
            has_unseen = ((work["cluster"] == c) & (~work["track_id"].isin(seen_ids))).any()
            if not has_unseen:
                sampled.append((c, -1.0))
            else:
                theta = rng.beta(a_c[c], b_c[c])
                sampled.append((c, float(theta)))

        sampled.sort(key=lambda x: x[1], reverse=True)
        if sampled[0][1] < 0:
            break  # niente unseen in nessun cluster

        chosen_cluster = sampled[0][0]
        pool = work[(work["cluster"] == chosen_cluster) & (~work["track_id"].isin(seen_ids))]
        if pool.empty:
            continue

        # scoring in-cluster
        max_pop_c = float(max_pop_by_cluster.get(chosen_cluster, max_pop_global) or 1.0)

        def row_score(r: pd.Series) -> float:
            pop = float(r.get("popularity", 0.0))
            pop_norm = pop / max_pop_c if max_pop_c > 0 else 0.0
            genre = _safe_str(r.get("track_genre", ""))
            artist = _safe_str(r.get("artists", ""))
            s = (
                cfg.w_pop * pop_norm
                + cfg.w_genre * get_genre_mean(genre)
                + cfg.w_artist * get_artist_mean(artist)
            )
            return float(s)

        # choose best score (deterministico)
        scores = pool.apply(row_score, axis=1)
        best_idx = scores.idxmax()
        row = pool.loc[best_idx]

        tid = str(row["track_id"])
        seen_ids.add(tid)

        reward, p_true = sim.rate(row)
        rewards.append(float(reward))
        p_trues.append(float(p_true))

        # update cluster beta
        if int(reward) == 1:
            a_c[chosen_cluster] += 1.0
        else:
            b_c[chosen_cluster] += 1.0

        # update genre beta
        genre = _safe_str(row.get("track_genre", "")).strip().lower()
        if genre != "":
            if genre not in a_g:
                a_g[genre], b_g[genre] = 1.0, 1.0
            if int(reward) == 1:
                a_g[genre] += 1.0
            else:
                b_g[genre] += 1.0

        # update artist beta
        artist = _safe_str(row.get("artists", "")).strip().lower()
        if artist != "":
            if artist not in a_a:
                a_a[artist], b_a[artist] = 1.0, 1.0
            if int(reward) == 1:
                a_a[artist] += 1.0
            else:
                b_a[artist] += 1.0

    return np.array(rewards, dtype=float), np.array(p_trues, dtype=float)


# -----------------------
# Aggregation + Plots
# -----------------------

def aggregate_and_plot(all_rewards: Dict[str, List[np.ndarray]], all_ptrues: Dict[str, List[np.ndarray]], T: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    algos = sorted(all_rewards.keys())

    # Per algo: matrice seeds x T (padded)
    rewards_mat = {}
    ptrues_mat = {}
    for a in algos:
        rewards_mat[a] = np.vstack([_pad_to_T(x, T) for x in all_rewards[a]])
        ptrues_mat[a] = np.vstack([_pad_to_T(x, T) for x in all_ptrues[a]])

    # mean per t (ignorando nan)
    mean_reward_t = {a: np.nanmean(rewards_mat[a], axis=0) for a in algos}
    mean_ptrues_t = {a: np.nanmean(ptrues_mat[a], axis=0) for a in algos}

    # cumulative like rate from mean rewards (ok per confronto)
    cum_like = {}
    exp_like = {}
    for a in algos:
        r = mean_reward_t[a].copy()
        p = mean_ptrues_t[a].copy()
        # se la serie finisce prima (nan), stop in quel punto
        valid = ~np.isnan(r)
        rr = r[valid]
        pp = p[valid]
        if len(rr) == 0:
            cum_like[a] = np.array([])
            exp_like[a] = np.array([])
            continue
        cum_like[a] = np.cumsum(rr) / (np.arange(len(rr)) + 1)
        exp_like[a] = np.cumsum(pp) / (np.arange(len(pp)) + 1)

    # rolling like rate
    roll_like = {a: _rolling_mean(cum_like[a] * 0 + mean_reward_t[a][~np.isnan(mean_reward_t[a])], 20) for a in algos}

    # Plot cumulative like rate
    plt.figure()
    for a in algos:
        y = cum_like[a]
        if len(y) == 0:
            continue
        plt.plot(y, label=a)
    plt.xlabel("t")
    plt.ylabel("Cumulative Like Rate (mean over seeds)")
    plt.title(f"Benchmark: cumulative like rate (T={T})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_like_rate_mean.png"))
    plt.close()

    # Plot rolling like rate (window=20)
    plt.figure()
    for a in algos:
        y = roll_like[a]
        if len(y) == 0:
            continue
        plt.plot(y, label=a)
    plt.xlabel("t")
    plt.ylabel("Rolling Like Rate (window=20, mean over seeds)")
    plt.title(f"Benchmark: rolling like rate (T={T}, window=20)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_like_rate_mean.png"))
    plt.close()

    # Plot expected like rate (da p_true)
    plt.figure()
    for a in algos:
        y = exp_like[a]
        if len(y) == 0:
            continue
        plt.plot(y, label=a)
    plt.xlabel("t")
    plt.ylabel("Expected Like Rate (mean p_true)")
    plt.title(f"Benchmark: expected like rate (T={T}, simulator ground truth)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expected_like_rate_mean.png"))
    plt.close()


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con canzoni + cluster (tracks_with_clusters.csv)")
    ap.add_argument("--T", type=int, default=2000, help="Numero interazioni nel loop per ogni seed")
    ap.add_argument("--seeds", type=int, default=20, help="Numero seed (ripetizioni)")
    ap.add_argument("--out", default="benchmark_results", help="Cartella output")
    ap.add_argument("--noise", type=float, default=0.15, help="Rumore del simulatore utente")

    ap.add_argument("--sim", choices=["full", "genre_only"], default="full", help="Tipo simulatore utente: full (feature+genre) o genre_only (solo macro-generi)")
    args = ap.parse_args()

    df = load_songs_csv(args.csv)
    feature_cols = ensure_feature_cols(df)

    # ensure columns exist
    if "popularity" not in df.columns:
        df["popularity"] = 0
    df["track_id"] = df["track_id"].astype(str)
    if "track_genre" not in df.columns:
        df["track_genre"] = ""

    algos = ["random", "popularity", "ts_hier", "cluster_ts_hybrid"]
    all_rewards: Dict[str, List[np.ndarray]] = {a: [] for a in algos}
    all_ptrues: Dict[str, List[np.ndarray]] = {a: [] for a in algos}

    raw_rows = []

    for seed in range(args.seeds):
        np.random.seed(seed)
        if args.sim == "genre_only":
            all_genres = df.get("track_genre", pd.Series([], dtype=str)).astype(str).unique().tolist()
            sim_cfg = GenreOnlyUserConfig(seed=seed, noise=float(args.noise))
            sim = GenreOnlyUserSimulator(all_genres=all_genres, config=sim_cfg)
        else:
            sim_cfg = UserSimConfig(seed=seed, noise=float(args.noise))
            sim = UserSimulator(df=df, feature_cols=feature_cols, config=sim_cfg)

        # random
        r, p = run_random(df.copy(), sim, args.T, seed=seed)
        all_rewards["random"].append(r); all_ptrues["random"].append(p)

        # popularity
        r, p = run_popularity(df.copy(), sim, args.T)
        all_rewards["popularity"].append(r); all_ptrues["popularity"].append(p)

        # ts_hier
        r, p = run_ts_hier(df.copy(), sim, feature_cols, args.T, seed=seed)
        all_rewards["ts_hier"].append(r); all_ptrues["ts_hier"].append(p)

        # cluster_ts_hybrid
        r, p = run_cluster_ts_hybrid(df.copy(), sim, args.T, seed=seed, cfg=HybridConfig())
        all_rewards["cluster_ts_hybrid"].append(r); all_ptrues["cluster_ts_hybrid"].append(p)

    # salva raw (lunghezze variabili)
    for algo in algos:
        for seed, (r, p) in enumerate(zip(all_rewards[algo], all_ptrues[algo])):
            for t in range(len(r)):
                raw_rows.append({"algo": algo, "seed": seed, "t": t, "reward": float(r[t]), "p_true": float(p[t])})

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(os.path.join(out_dir, "raw_logs.csv"), index=False)

    aggregate_and_plot(all_rewards, all_ptrues, T=int(args.T), out_dir=out_dir)

    print(f"Benchmark completato. T={args.T}, seeds={args.seeds}, sim={args.sim}")
    print("Output:", out_dir)
    print("File:", "raw_logs.csv, cumulative_like_rate_mean.png, rolling_like_rate_mean.png, expected_like_rate_mean.png")


if __name__ == "__main__":
    main()
