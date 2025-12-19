"""
RUN ONLY TS (v3): esegue Cold Start + FullHierarchicalTS in modalità AUTO (simulatore utente),
salva log + grafici in una cartella nuova (non sovrascrive mai).

Usa ESCLUSIVAMENTE i simulatori definiti in sim_users_v3.py.

Uso esempi:
  python run_only_ts_v3.py --csv tracks_with_clusters.csv --user feature
  python run_only_ts_v3.py --csv tracks_with_clusters.csv --user feature_shock --shock_at 1000
  python run_only_ts_v3.py --csv tracks_with_clusters.csv --user macro
  python run_only_ts_v3.py --csv tracks_with_clusters.csv --user macro_shock --shock_at 800
  python run_only_ts_v3.py --csv tracks_with_clusters.csv --user random

Note:
- Seed fisso interno (42). Non serve passarne uno da CLI.
- Ogni run crea una nuova cartella in results_only_ts/.
"""

from __future__ import annotations

#import sys
from pathlib import Path

# aggiunge: .../grupoX-Final-Exam_Spotify/Tentativo al PYTHONPATH
#TENTATIVO = Path(__file__).resolve().parents[1] / "Tentativo"
#sys.path.insert(0, str(TENTATIVO))

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_utils import load_songs_csv, ensure_feature_cols
from cold_start_clustered import build_cold_start, ask_labels
try:
    from hierarchical_ts_genre_forgetting import FullHierarchicalTS  # type: ignore
except Exception:
    print("ATTENZIONE")
from performance import Tracker

# === USERS (v3) ===
from sim_users_v3 import (
    FeatureUserSimulator, FeatureUserConfig,
    MacroGenreProbUserSimulator, MacroGenreProbUserConfig,
    RandomUserSimulator, RandomUserConfig,
)


def _make_run_dir(base_out: str) -> str:
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{np.random.default_rng().integers(0, 1_000_000):06d}"
    run_dir = base / f"run_{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def format_song_row(row: pd.Series) -> str:
    return (
        f"{row.get('track_name','')} — {row.get('artists','')}"
        f" | genre={row.get('track_genre','')}"
        f" | pop={row.get('popularity','')}"
        f" | cluster={row.get('cluster','')}"
    )


def build_simulator(args, df: pd.DataFrame, feature_cols: list[str]):
    # seed fisso interno
    seed = 42
    np.random.seed(seed)

    if args.user == "feature":
        cfg = FeatureUserConfig(
            seed=seed,
            noise=args.noise,
            shock_mode="none",
        )
        return FeatureUserSimulator(df, feature_cols, config=cfg)

    if args.user == "feature_shock":
        cfg = FeatureUserConfig(
            seed=seed,
            noise=args.noise,
            shock_mode="hard",
            shock_at=args.shock_at,
        )
        return FeatureUserSimulator(df, feature_cols, config=cfg)

    if args.user == "feature_gradual":
        cfg = FeatureUserConfig(
            seed=seed,
            noise=args.noise,
            shock_mode="gradual",
            shock_at=args.shock_at,
            shock_window=args.shock_window,
        )
        return FeatureUserSimulator(df, feature_cols, config=cfg)

    if args.user == "macro":
        cfg = MacroGenreProbUserConfig(seed=seed, shock_mode="none")
        all_genres = df["track_genre"].astype(str).tolist()
        return MacroGenreProbUserSimulator(all_genres=all_genres, config=cfg)

    if args.user == "macro_shock":
        cfg = MacroGenreProbUserConfig(
            seed=seed,
            shock_mode="hard",
            shock_at=args.shock_at,
            shock_type=args.macro_shock_type,
        )
        all_genres = df["track_genre"].astype(str).tolist()
        return MacroGenreProbUserSimulator(all_genres=all_genres, config=cfg)

    if args.user == "random":
        return RandomUserSimulator(RandomUserConfig(seed=seed))

    raise ValueError(f"User type non supportato: {args.user}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path al file tracks_with_clusters.csv")
    ap.add_argument("--out", default="results_only_ts", help="Cartella base di output")
    ap.add_argument("--max_steps", type=int, default=2000, help="Max raccomandazioni nel loop")

    ap.add_argument(
        "--user",
        required=True,
        choices=[
            "feature",
            "feature_shock",
            "feature_gradual",
            "macro",
            "macro_shock",
            "random",
        ],
        help="Tipo di simulatore utente (v3)",
    )

    ap.add_argument("--shock_at", type=int, default=1000, help="Iterazione di inizio shock")
    ap.add_argument("--shock_window", type=int, default=200, help="Finestra shock graduale")
    ap.add_argument("--macro_shock_type", choices=["invert", "reroll"], default="invert")
    ap.add_argument("--noise", type=float, default=0.0, help="Rumore del simulatore")

    args = ap.parse_args()

    # --- Load data ---
    df = load_songs_csv(args.csv)
    feature_cols = ensure_feature_cols(df)

    # --- Simulator ---
    sim = build_simulator(args, df=df, feature_cols=feature_cols)

    # --- Cold start ---
    cold_items = build_cold_start(df, n_per_cluster=2)
    labeled = ask_labels(cold_items, df=df, auto_rater=sim)

    if len(labeled) == 0:
        print("Cold start senza label: termino.")
        return

    user_history = pd.DataFrame(labeled)

    # --- Init TS ---
    ts = FullHierarchicalTS(df_songs=df, feature_cols=feature_cols, seed=42)
    ts.initialize_from_cold_start(user_history=user_history, label_col="label")

    tracker = Tracker()

    # log cold start
    t = 0
    for it in labeled:
        tid = str(it["track_id"])
        row = df[df["track_id"].astype(str) == tid].iloc[0]
        cluster = int(row["cluster"])
        p_pred = ts.predict_like_probability(track_id=tid, cluster_idx=None)
        tracker.log(t=t, track_id=tid, cluster=cluster, p_pred=float(p_pred), reward=int(it["label"]))
        t += 1

    # --- Loop TS ---
    print("\n=== Inizio raccomandazioni (RUN ONLY TS v3) ===\n")
    for step in range(args.max_steps):
        track_id, cluster_idx = ts.recommend_one()
        if track_id is None:
            print("Candidate pool esaurito.")
            break

        row = df[df["track_id"].astype(str) == str(track_id)].iloc[0]
        p_pred = float(ts.predict_like_probability(track_id=str(track_id), cluster_idx=cluster_idx))

        sim.set_turn(step)
        reward, p_true = sim.rate(row)

        print(
            f"[{step+1}] {format_song_row(row)} | "
            f"p_pred={p_pred:.3f} | r={reward} | p_true={p_true:.3f}"
        )

        ts.update(track_id=str(track_id), reward=int(reward))
        tracker.log(t=t, track_id=str(track_id), cluster=int(row["cluster"]), p_pred=p_pred, reward=int(reward))
        t += 1

    # --- Output ---
    run_dir = _make_run_dir(args.out)
    log_path = tracker.save(run_dir)
    plot_paths = tracker.plot_all(run_dir)

    print("\n=== Fine run ===")
    print("Cartella run:", run_dir)
    print("Log salvato:", log_path)
    for name, pth in plot_paths.items():
        print(f"Grafico '{name}' salvato:", pth)


if __name__ == "__main__":
    main()
