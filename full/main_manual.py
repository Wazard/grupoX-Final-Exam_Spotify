"""
MAIN MANUALE: sessione interattiva per raccomandazioni musicali
Cold Start + FullHierarchicalTS con feedback umano (1/0/q).

Scopo:
- demo qualitativa
- debugging del TS
- ispezione delle raccomandazioni

NOTE IMPORTANTI:
- NON è pensato per benchmark o grafici comparabili
- usa un seed fisso interno (42)
- salva comunque log + grafici (una nuova cartella ogni run)

Uso:
  python main_manual.py --csv tracks_with_clusters.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_songs_csv, ensure_feature_cols
from cold_start_clustered import build_cold_start, ask_labels
from hierarchical_ts_genre_forgetting import FullHierarchicalTS
from performance import Tracker

# users (solo per cold start automatico opzionale)
from sim_users_v3 import RandomUserSimulator, RandomUserConfig


def _make_run_dir(base_out: str) -> str:
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"manual_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def format_song_row(row: pd.Series) -> str:
    return (
        f"{row.get('track_name','')} — {row.get('artists','')}"
        f" | genre={row.get('track_genre','')}"
        f" | pop={row.get('popularity','')}"
        f" | cluster={row.get('cluster','')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path al file tracks_with_clusters.csv")
    ap.add_argument("--out", default="results_manual", help="Cartella di output")
    ap.add_argument("--max_steps", type=int, default=2000, help="Max raccomandazioni")
    ap.add_argument(
        "--auto_cold_start",
        action="store_true",
        help="Usa un random user per il cold start (altrimenti chiedi input manuale)",
    )
    args = ap.parse_args()

    # seed fisso interno
    seed = 42
    np.random.seed(seed)

    # --- Load data ---
    df = load_songs_csv(args.csv)
    feature_cols = ensure_feature_cols(df)

    # --- Cold start ---
    print("\n=== COLD START ===\n")
    if args.auto_cold_start:
        auto_user = RandomUserSimulator(RandomUserConfig(seed=seed))
    else:
        auto_user = None

    cold_items = build_cold_start(df, n_per_cluster=2)
    labeled = ask_labels(cold_items, df=df, auto_rater=auto_user)

    if len(labeled) == 0:
        print("Cold start senza label: termino.")
        return

    user_history = pd.DataFrame(labeled)

    # --- Init TS ---
    ts = FullHierarchicalTS(df_songs=df, feature_cols=feature_cols, seed=seed)
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

    # --- Loop manuale ---
    print("\n=== INIZIO RACCOMANDAZIONI (MANUALE) ===\n")
    for step in range(args.max_steps):
        track_id, cluster_idx = ts.recommend_one()
        if track_id is None:
            print("Candidate pool esaurito.")
            break

        row = df[df["track_id"].astype(str) == str(track_id)].iloc[0]
        p_pred = float(ts.predict_like_probability(track_id=str(track_id), cluster_idx=cluster_idx))

        print(f"[{step+1}] {format_song_row(row)}")
        print(f"Probabilità stimata che ti piaccia: {p_pred:.3f}")

        while True:
            s = input("Ti piace? (1=si / 0=no / q=quit): ").strip().lower()
            if s == "q":
                print("Uscita manuale.")
                step = args.max_steps
                break
            if s in {"0", "1"}:
                reward = int(s)
                break
            print("Input non valido.")

        ts.update(track_id=str(track_id), reward=int(reward))
        tracker.log(t=t, track_id=str(track_id), cluster=int(row["cluster"]), p_pred=p_pred, reward=int(reward))
        t += 1
        print()

    # --- Output ---
    run_dir = _make_run_dir(args.out)
    log_path = tracker.save(run_dir)
    plot_paths = tracker.plot_all(run_dir)

    print("\n=== FINE SESSIONE MANUALE ===")
    print("Cartella run:", run_dir)
    print("Log salvato:", log_path)
    for name, pth in plot_paths.items():
        print(f"Grafico '{name}' salvato:", pth)


if __name__ == "__main__":
    main()
