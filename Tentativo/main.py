"""
MAIN: sessione (manuale o automatica) per raccomandazioni musicali con
Cold Start per cluster + Hierarchical Thompson Sampling.

MODALITÀ:
- Imposta AUTO = True  -> simulatore utente (nessun input da tastiera)
- Imposta AUTO = False -> modalità manuale (chiede 1/0/q)

Uso:
  python main.py --csv tracks_with_clusters.csv

Note:
- non ripropone mai una canzone già vista
- salva log e grafici in runs/run_seedX/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data_utils import load_songs_csv, ensure_feature_cols
from cold_start_clustered import build_cold_start, ask_labels
from hierarchical_ts import FullHierarchicalTS
from performance import Tracker

from sim_user import UserSimulator, UserSimConfig

# ==========================
# SWITCH CHIARO AUTO/MANUALE
# ==========================
AUTO = True  # <-- metti False se vuoi la modalità manuale


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
    ap.add_argument("--out", default="runs", help="Cartella di output (log + grafici)")
    ap.add_argument("--seed", type=int, default=42, help="Seed random per riproducibilità")
    ap.add_argument("--max_steps", type=int, default=2000, help="Max raccomandazioni nel loop")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # --- Load data ---
    df = load_songs_csv(args.csv)
    feature_cols = ensure_feature_cols(df)

    # --- Simulator (opzionale) ---
    sim = None
    if AUTO:
        # utente coerente: preferenze su feature + bias per track_genre (NO cluster bias)
        sim_cfg = UserSimConfig(seed=args.seed, noise=0.15)
        sim = UserSimulator(df=df, feature_cols=feature_cols, config=sim_cfg)

    # --- Cold start ---
    cold_items = build_cold_start(df, n_per_cluster=2)
    labeled = ask_labels(cold_items, df=df, auto_rater=sim)

    if len(labeled) == 0:
        print("Cold start senza label: termino.")
        return

    user_history = pd.DataFrame(labeled)

    # --- Init TS ---
    ts = FullHierarchicalTS(df_songs=df, feature_cols=feature_cols, seed=args.seed)
    ts.initialize_from_cold_start(user_history=user_history, label_col="label")

    tracker = Tracker()

    # log anche cold start (con p_pred dopo init)
    t = 0
    for it in labeled:
        tid = str(it["track_id"])
        row = df[df["track_id"].astype(str) == tid].iloc[0]
        cluster = int(row["cluster"])
        p_pred = ts.predict_like_probability(track_id=tid, cluster_idx=None)
        tracker.log(t=t, track_id=tid, cluster=cluster, p_pred=float(p_pred), reward=int(it["label"]))
        t += 1

    # --- Loop ---
    print("\n=== Inizio raccomandazioni ===\n")
    for step in range(args.max_steps):
        track_id, cluster_idx = ts.recommend_one()
        if track_id is None:
            print("Nessuna canzone disponibile (candidate pool esaurito).")
            break

        row = df[df["track_id"].astype(str) == str(track_id)].iloc[0]
        p_pred = float(ts.predict_like_probability(track_id=str(track_id), cluster_idx=cluster_idx))

        print(f"[{step+1}] {format_song_row(row)}")
        print(f"Probabilità stimata che ti piaccia: {p_pred:.3f}")

        if AUTO:
            assert sim is not None
            reward, p_true = sim.rate(row)
            print(f"(AUTO) feedback: {reward} | p_like_true={p_true:.3f}")
        else:
            while True:
                s = input("Ti piace? (1/0/q): ").strip().lower()
                if s == "q":
                    reward = None
                    break
                if s in {"0", "1"}:
                    reward = int(s)
                    break
                print("Input non valido: usa 1, 0 oppure q.")
            if reward is None:
                break

        ts.update(track_id=str(track_id), reward=int(reward))
        tracker.log(t=t, track_id=str(track_id), cluster=int(row["cluster"]), p_pred=p_pred, reward=int(reward))
        t += 1
        print()

    # --- Output ---
    run_dir = os.path.join(args.out, f"run_seed{args.seed}")
    log_path = tracker.save(run_dir)
    plot_paths = tracker.plot_all(run_dir)

    print("\n=== Fine sessione ===")
    print("Log salvato:", log_path)
    for name, pth in plot_paths.items():
        print(f"Grafico '{name}' salvato:", pth)


if __name__ == "__main__":
    main()
