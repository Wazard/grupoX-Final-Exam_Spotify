"""
Utility per caricamento dati e gestione feature.
"""

from __future__ import annotations
import pandas as pd


# Feature numeriche usate dal modello
DEFAULT_FEATURE_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
]


def load_songs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # sicurezza: track_id come stringa
    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].astype(str)

    return df


def ensure_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Verifica quali feature numeriche sono presenti nel DataFrame.
    Ritorna la lista effettivamente usabile.
    """
    available = []
    for c in DEFAULT_FEATURE_COLS:
        if c in df.columns:
            available.append(c)

    if len(available) == 0:
        raise ValueError("Nessuna feature numerica trovata nel CSV.")

    return available
