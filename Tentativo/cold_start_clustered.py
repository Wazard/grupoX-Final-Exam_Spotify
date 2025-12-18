"""
Cold start per cluster:
- sceglie n_per_cluster canzoni per ogni cluster (default 2)
- ranking intra-cluster basato su popolarità (condizionata al cluster)
- evita ripetizioni di artista (soft)
- opzionalmente etichetta automaticamente usando un simulatore (auto_rater)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def build_cold_start(df: pd.DataFrame, n_per_cluster: int = 2) -> List[Dict]:
    """
    Ritorna una lista di dict con track_id (e cluster) per il cold start.
    Se un cluster ha poche canzoni, prende quelle disponibili.
    Se mancano colonne, prova comunque a funzionare.
    """
    if "cluster" not in df.columns:
        raise ValueError("La colonna 'cluster' non è presente nel CSV.")

    # copia di lavoro
    work = df.copy()

    # popolarità: se manca, metti 0
    if "popularity" not in work.columns:
        work["popularity"] = 0

    # ordina per cluster poi popolarità desc
    work = work.sort_values(["cluster", "popularity"], ascending=[True, False])

    chosen = []
    used_artists_global = set()

    for c, g in work.groupby("cluster", sort=True):
        picked = 0
        used_artists_local = set()

        # prova a scegliere canzoni con artisti diversi (prima locale, poi globale)
        for _, row in g.iterrows():
            if picked >= n_per_cluster:
                break

            tid = str(row.get("track_id", ""))
            if tid == "":
                continue

            artists = str(row.get("artists", "")).strip().lower()

            # soft constraints
            if artists and artists in used_artists_local:
                continue
            if artists and artists in used_artists_global and (picked == 0):
                # per la prima del cluster cerchiamo di evitare artisti già visti globalmente
                continue

            chosen.append({"track_id": tid, "cluster": int(c)})
            picked += 1
            if artists:
                used_artists_local.add(artists)
                used_artists_global.add(artists)

        # fallback: se non hai trovato abbastanza per i vincoli artisti, riempi con le top rimanenti
        if picked < n_per_cluster:
            for _, row in g.iterrows():
                if picked >= n_per_cluster:
                    break
                tid = str(row.get("track_id", ""))
                if tid == "":
                    continue
                if any(x["track_id"] == tid for x in chosen):
                    continue
                chosen.append({"track_id": tid, "cluster": int(c)})
                picked += 1

    return chosen


def ask_labels(cold_items: List[Dict], df: pd.DataFrame, auto_rater=None) -> List[Dict]:
    """
    cold_items: lista di dict con track_id
    auto_rater: oggetto con metodo rate(row) -> (label, p_true)
    Ritorna: lista dict {track_id, label}
    """
    labeled: List[Dict] = []

    # indicizza per velocità
    id_to_row = {str(rid): i for i, rid in enumerate(df["track_id"].astype(str).tolist())}

    for it in cold_items:
        tid = str(it["track_id"])

        if tid not in id_to_row:
            # se non lo trovo, skippo
            continue

        row = df.iloc[id_to_row[tid]]

        if auto_rater is not None:
            label, p_true = auto_rater.rate(row)
            print(
                f"(AUTO-COLD) {row.get('track_name','')} — {row.get('artists','')} "
                f"| genre={row.get('track_genre','')} | pop={row.get('popularity','')} "
                f"=> label={label} (p_true={p_true:.3f})"
            )
        else:
            print(
                f"{row.get('track_name','')} — {row.get('artists','')} "
                f"| genre={row.get('track_genre','')} | pop={row.get('popularity','')} "
                f"| cluster={row.get('cluster','')}"
            )
            while True:
                s = input("Ti è piaciuta? (1/0): ").strip()
                if s in {"0", "1"}:
                    label = int(s)
                    break
                print("Input non valido: usa 1 oppure 0.")

        labeled.append({"track_id": tid, "label": int(label)})

    return labeled
