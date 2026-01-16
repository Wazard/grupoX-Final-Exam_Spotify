
"""
Benchmark v4 for the music recommender.

Adds:
- Regret plots (cumulative regret and average regret) computed from observed rewards.
  Here regret is defined as: r_t = 1 - reward_t (so it measures "missed likes").
  This matches the practical bandit notion when rewards are binary and max reward is 1.
- Estimate regret growth exponent alpha by fitting log(cum_regret) ~ alpha * log(t).

Simulators (choose via --sim):
- feature_stationary
- feature_shock_hard
- feature_shock_gradual
- macro_prob_stationary
- macro_prob_shock_hard_invert
- macro_prob_shock_gradual_invert
- random_05

Algorithms compared:
- random
- popularity
- ts_hier (your hierarchical TS: cluster TS + linear TS (songs), with your updated hierarchical_ts file)
- cluster_ts_hybrid (cluster TS + deterministic in-cluster scoring)

Usage:
  python benchmark_v4.py --csv tracks_with_clusters.csv --T 2000 --seeds 20 --sim feature_shock_hard --out benchmark_results

Notes:
- Unique run directory is always created (no overwrite).
- You MUST call sim.set_turn(t) before sim.rate(row) to enable shocks.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from data_utils import load_songs_csv, ensure_feature_cols
from cold_start_clustered import build_cold_start, ask_labels

# Try importing the updated TS implementation first
try:
    from hierarchical_ts_genre_forgetting import FullHierarchicalTS  # type: ignore
except Exception:
    from hierarchical_ts import FullHierarchicalTS  # type: ignore

from sim_users_v4 import (
    FeatureUserSimulator, FeatureUserConfig,
    MacroGenreProbUserSimulator, MacroGenreProbUserConfig,
    RealisticUserSimulator, RealisticUserConfig,
    RandomUserSimulator, RandomUserConfig,
)


# -----------------------
# Run dir
# -----------------------

def make_unique_run_dir(base_out: str, *, sim: str, T: int, seeds: int, run_name: Optional[str] = None) -> str:
    os.makedirs(base_out, exist_ok=True)
    if run_name and run_name.strip():
        sub = run_name.strip()
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sub = f"run_{ts}_sim-{sim}_T-{T}_seeds-{seeds}"
    out_dir = os.path.join(base_out, sub)
    if os.path.exists(out_dir):
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
        out_dir = out_dir + "_" + suffix
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# -----------------------
# Util
# -----------------------

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


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def _estimate_regret_exponent(cum_reg: np.ndarray, *, min_t: int = 50) -> float:
    """
    Fit log(cum_reg(t)) = a + alpha log(t) on t>=min_t.
    Returns alpha. If not enough points, returns NaN.
    """
    y = np.asarray(cum_reg, dtype=float)
    if len(y) <= min_t:
        return float("nan")
    t = np.arange(1, len(y) + 1, dtype=float)
    mask = (t >= float(min_t)) & np.isfinite(y) & (y > 0)
    if mask.sum() < 10:
        return float("nan")
    X = np.log(t[mask])
    Y = np.log(y[mask])
    # simple least squares slope
    alpha = float(np.cov(X, Y, bias=True)[0, 1] / (np.var(X) + 1e-12))
    return alpha


# -----------------------
# Runner: Random
# -----------------------

def run_random(df: pd.DataFrame, sim, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unseen = df.copy()
    rewards, p_trues = [], []

    for t in range(T):
        if unseen.empty:
            break
        sim.set_turn(t)
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
    for t in range(T):
        if unseen.empty:
            break
        sim.set_turn(t)
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
    cold = build_cold_start(df, n_per_cluster=2)
    labeled = ask_labels(cold, df=df, auto_rater=sim)

    user_hist = pd.DataFrame(labeled)
    ts = FullHierarchicalTS(df_songs=df, feature_cols=feature_cols, seed=seed)
    ts.initialize_from_cold_start(user_history=user_hist, label_col="label")

    rewards, p_trues = [], []
    for t in range(T):
        track_id, _cluster_idx = ts.recommend_one()
        if track_id is None:
            break
        row = df[df["track_id"].astype(str) == str(track_id)].iloc[0]
        sim.set_turn(t)
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

    a_c = {c: 1.0 for c in clusters}
    b_c = {c: 1.0 for c in clusters}

    a_g: Dict[str, float] = {}
    b_g: Dict[str, float] = {}
    a_a: Dict[str, float] = {}
    b_a: Dict[str, float] = {}

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

    for t in range(T):
        sampled = []
        for c in clusters:
            has_unseen = ((work["cluster"] == c) & (~work["track_id"].isin(seen_ids))).any()
            if not has_unseen:
                sampled.append((c, -1.0))
            else:
                theta = rng.beta(a_c[c], b_c[c])
                sampled.append((c, float(theta)))

        sampled.sort(key=lambda x: x[1], reverse=True)
        if sampled[0][1] < 0:
            break

        chosen_cluster = sampled[0][0]
        pool = work[(work["cluster"] == chosen_cluster) & (~work["track_id"].isin(seen_ids))]
        if pool.empty:
            continue

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

        scores = pool.apply(row_score, axis=1)
        best_idx = scores.idxmax()
        row = pool.loc[best_idx]

        tid = str(row["track_id"])
        seen_ids.add(tid)

        sim.set_turn(t)
        reward, p_true = sim.rate(row)
        rewards.append(float(reward))
        p_trues.append(float(p_true))

        if int(reward) == 1:
            a_c[chosen_cluster] += 1.0
        else:
            b_c[chosen_cluster] += 1.0

        genre = _safe_str(row.get("track_genre", "")).strip().lower()
        if genre != "":
            if genre not in a_g:
                a_g[genre], b_g[genre] = 1.0, 1.0
            if int(reward) == 1:
                a_g[genre] += 1.0
            else:
                b_g[genre] += 1.0

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

def aggregate_and_plot(all_rewards: Dict[str, List[np.ndarray]],
                       all_ptrues: Dict[str, List[np.ndarray]],
                       T: int,
                       out_dir: str,
                       rolling_w: int = 20) -> None:
    os.makedirs(out_dir, exist_ok=True)
    algos = sorted(all_rewards.keys())

    rewards_mat = {a: np.vstack([_pad_to_T(x, T) for x in all_rewards[a]]) for a in algos}
    ptrues_mat  = {a: np.vstack([_pad_to_T(x, T) for x in all_ptrues[a]]) for a in algos}

    # --- Like-rate plots (same idea as before) ---
    mean_reward_t = {a: np.nanmean(rewards_mat[a], axis=0) for a in algos}
    mean_ptrues_t = {a: np.nanmean(ptrues_mat[a], axis=0) for a in algos}

    cum_like = {}
    exp_like = {}
    roll_like = {}
    for a in algos:
        r = mean_reward_t[a]
        p = mean_ptrues_t[a]
        valid = ~np.isnan(r)
        rr = r[valid]
        pp = p[valid]
        if len(rr) == 0:
            cum_like[a] = np.array([])
            exp_like[a] = np.array([])
            roll_like[a] = np.array([])
            continue
        cum_like[a] = np.cumsum(rr) / (np.arange(len(rr)) + 1)
        exp_like[a] = np.cumsum(pp) / (np.arange(len(pp)) + 1)
        roll_like[a] = _rolling_mean(rr, rolling_w)

    plt.figure()
    for a in algos:
        if len(cum_like[a]) == 0:
            continue
        plt.plot(cum_like[a], label=a)
    plt.xlabel("t")
    plt.ylabel("Cumulative Like Rate (mean over seeds)")
    plt.title(f"Cumulative like rate (T={T})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_like_rate_mean.png"))
    plt.close()

    plt.figure()
    for a in algos:
        if len(roll_like[a]) == 0:
            continue
        plt.plot(roll_like[a], label=a)
    plt.xlabel("t")
    plt.ylabel(f"Rolling Like Rate (window={rolling_w}, mean over seeds)")
    plt.title(f"Rolling like rate (T={T}, window={rolling_w})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_like_rate_mean.png"))
    plt.close()

    plt.figure()
    for a in algos:
        if len(exp_like[a]) == 0:
            continue
        plt.plot(exp_like[a], label=a)
    plt.xlabel("t")
    plt.ylabel("Expected Like Rate (mean p_true)")
    plt.title(f"Expected like rate (T={T}, simulator ground truth)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expected_like_rate_mean.png"))
    plt.close()

    # --- Regret (per-seed, then average) ---
    cum_reg_mean = {}
    avg_reg_mean = {}
    alpha_est = {}

    for a in algos:
        # per seed cumulative regret: sum(1 - reward)
        mats = rewards_mat[a]
        per_seed = []
        for s in range(mats.shape[0]):
            r = mats[s, :]
            valid = ~np.isnan(r)
            rr = r[valid]
            if len(rr) == 0:
                per_seed.append(np.array([]))
                continue
            inst_reg = 1.0 - rr
            per_seed.append(np.cumsum(inst_reg))
        # pad and mean
        per_seed_pad = np.vstack([_pad_to_T(x, T) for x in per_seed])
        mean_cum_reg = np.nanmean(per_seed_pad, axis=0)
        # trim trailing NaNs
        valid = ~np.isnan(mean_cum_reg)
        mean_cum_reg = mean_cum_reg[valid]
        cum_reg_mean[a] = mean_cum_reg
        if len(mean_cum_reg) > 0:
            t = np.arange(1, len(mean_cum_reg) + 1)
            avg_reg_mean[a] = mean_cum_reg / t
            alpha_est[a] = _estimate_regret_exponent(mean_cum_reg, min_t=max(50, rolling_w))
        else:
            avg_reg_mean[a] = np.array([])
            alpha_est[a] = float("nan")

    plt.figure()
    for a in algos:
        if len(cum_reg_mean[a]) == 0:
            continue
        plt.plot(cum_reg_mean[a], label=a)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret (mean over seeds), r_t = 1 - reward")
    plt.title(f"Cumulative regret (T={T})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_regret_mean.png"))
    plt.close()

    plt.figure()
    for a in algos:
        if len(avg_reg_mean[a]) == 0:
            continue
        plt.plot(avg_reg_mean[a], label=a)
    plt.xlabel("t")
    plt.ylabel("Average Regret (mean cum_reg / t)")
    plt.title(f"Average regret (T={T})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_regret_mean.png"))
    plt.close()

    # Save exponent estimates
    df_alpha = pd.DataFrame([{"algo": a, "alpha_hat": alpha_est[a]} for a in algos]).sort_values("algo")
    df_alpha.to_csv(os.path.join(out_dir, "regret_exponent_alpha_hat.csv"), index=False)


# -----------------------
# Sim factory
# -----------------------

def make_simulator(sim_name: str, df: pd.DataFrame, feature_cols: List[str], seed: int, noise: float,
                   shock_at: int, shock_window: int) -> object:
    sim_name = str(sim_name)

    if sim_name == "feature_stationary":
        cfg = FeatureUserConfig(seed=seed, noise=float(noise), shock_mode="none")
        return FeatureUserSimulator(df=df, feature_cols=feature_cols, config=cfg)

    if sim_name == "feature_shock_hard":
        cfg = FeatureUserConfig(seed=seed, noise=float(noise), shock_mode="hard", shock_at=int(shock_at))
        return FeatureUserSimulator(df=df, feature_cols=feature_cols, config=cfg)

    if sim_name == "feature_shock_gradual":
        cfg = FeatureUserConfig(seed=seed, noise=float(noise), shock_mode="gradual", shock_at=int(shock_at), shock_window=int(shock_window))
        return FeatureUserSimulator(df=df, feature_cols=feature_cols, config=cfg)

    all_genres = df.get("track_genre", pd.Series([], dtype=str)).astype(str).unique().tolist()

    if sim_name == "macro_prob_stationary":
        cfg = MacroGenreProbUserConfig(seed=seed, noise=float(noise), shock_mode="none")
        return MacroGenreProbUserSimulator(all_genres=all_genres, config=cfg)

    if sim_name == "macro_prob_shock_hard_invert":
        cfg = MacroGenreProbUserConfig(seed=seed, noise=float(noise), shock_mode="hard", shock_at=int(shock_at), shock_type="invert")
        return MacroGenreProbUserSimulator(all_genres=all_genres, config=cfg)

    if sim_name == "macro_prob_shock_gradual_invert":
        cfg = MacroGenreProbUserConfig(seed=seed, noise=float(noise), shock_mode="gradual", shock_at=int(shock_at), shock_window=int(shock_window), shock_type="invert")
        return MacroGenreProbUserSimulator(all_genres=all_genres, config=cfg)

    if sim_name == "realistic_user":
        cfg = RealisticUserConfig(seed=seed, noise=float(noise))
        return RealisticUserSimulator(df=df, feature_cols=feature_cols, config=cfg)

    if sim_name == "random_05":
        cfg = RandomUserConfig(seed=seed, p=0.5)
        return RandomUserSimulator(config=cfg)

    raise ValueError(f"Unknown sim '{sim_name}'")


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con canzoni + cluster (tracks_with_clusters.csv)")
    ap.add_argument("--T", type=int, default=2000, help="Numero interazioni nel loop per ogni seed")
    ap.add_argument("--seeds", type=int, default=20, help="Numero seed (ripetizioni)")
    ap.add_argument("--out", default="benchmark_results", help="Cartella output")
    ap.add_argument("--run_name", default=None, help="Nome cartella run (opzionale). Se omesso usa timestamp.")
    ap.add_argument("--noise", type=float, default=0.15, help="Rumore del simulatore (interpretato per sim scelto)")

    ap.add_argument("--sim", default="feature_stationary",
                    choices=[
                        "feature_stationary", "feature_shock_hard", "feature_shock_gradual",
                        "macro_prob_stationary", "macro_prob_shock_hard_invert", "macro_prob_shock_gradual_invert",
                        "realistic_user",
                        "random_05",
                    ],
                    help="Tipo simulatore utente")

    ap.add_argument("--shock_at", type=int, default=1000, help="Turno in cui inizia lo shock (se attivo)")
    ap.add_argument("--shock_window", type=int, default=200, help="Durata shock graduale (se attivo)")
    ap.add_argument("--rolling_w", type=int, default=20, help="Window rolling per i grafici")
    args = ap.parse_args()

    df = load_songs_csv(args.csv)
    feature_cols = ensure_feature_cols(df)

    if "popularity" not in df.columns:
        df["popularity"] = 0
    df["track_id"] = df["track_id"].astype(str)
    if "track_genre" not in df.columns:
        df["track_genre"] = ""

    algos = ["random", "popularity", "ts_hier", "cluster_ts_hybrid"]
    all_rewards: Dict[str, List[np.ndarray]] = {a: [] for a in algos}
    all_ptrues: Dict[str, List[np.ndarray]] = {a: [] for a in algos}

    raw_rows = []

    for seed in tqdm(range(int(args.seeds)), desc="Running seeds", unit="seed"):
        # simulator
        sim = make_simulator(
            sim_name=str(args.sim),
            df=df,
            feature_cols=feature_cols,
            seed=seed,
            noise=float(args.noise),
            shock_at=int(args.shock_at),
            shock_window=int(args.shock_window),
        )

        # random
        r, p = run_random(df.copy(), sim, int(args.T), seed=seed)
        all_rewards["random"].append(r); all_ptrues["random"].append(p)

        # popularity
        r, p = run_popularity(df.copy(), sim, int(args.T))
        all_rewards["popularity"].append(r); all_ptrues["popularity"].append(p)

        # ts_hier
        r, p = run_ts_hier(df.copy(), sim, feature_cols, int(args.T), seed=seed)
        all_rewards["ts_hier"].append(r); all_ptrues["ts_hier"].append(p)

        # cluster_ts_hybrid
        r, p = run_cluster_ts_hybrid(df.copy(), sim, int(args.T), seed=seed, cfg=HybridConfig())
        all_rewards["cluster_ts_hybrid"].append(r); all_ptrues["cluster_ts_hybrid"].append(p)

    # raw logs
    for algo in algos:
        for seed, (r, p) in enumerate(zip(all_rewards[algo], all_ptrues[algo])):
            for t in range(len(r)):
                raw_rows.append({"algo": algo, "seed": seed, "t": t, "reward": float(r[t]), "p_true": float(p[t])})

    out_dir = make_unique_run_dir(args.out, sim=str(args.sim), T=int(args.T), seeds=int(args.seeds), run_name=args.run_name)
    pd.DataFrame(raw_rows).to_csv(os.path.join(out_dir, "raw_logs.csv"), index=False)

    aggregate_and_plot(all_rewards, all_ptrues, T=int(args.T), out_dir=out_dir, rolling_w=int(args.rolling_w))

    print(f"Benchmark completato. T={args.T}, seeds={args.seeds}, sim={args.sim}, noise={args.noise}")
    print("Output:", out_dir)
    print("Files: raw_logs.csv, cumulative_like_rate_mean.png, rolling_like_rate_mean.png, expected_like_rate_mean.png,")
    print("       cumulative_regret_mean.png, avg_regret_mean.png, regret_exponent_alpha_hat.csv")


if __name__ == "__main__":
    main()
