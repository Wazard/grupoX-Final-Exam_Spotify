"""
Config centrale del progetto "AI DJ".
"""

from dataclasses import dataclass
from typing import List


# Colonne necessarie nel CSV
REQUIRED_COLS = [
    "track_id", "artists", "track_name", "album_name",
    "popularity", "track_genre", "cluster"
]

# Feature numeriche usate dal Linear Thompson Sampling (evito colonne testuali/target)
DEFAULT_FEATURE_COLS: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Parametri cold start
POPULARITY_THRESHOLD = 25
COLDSTART_SONGS_PER_CLUSTER = 2

# Parametri plotting
ROLLING_WINDOW = 20


@dataclass(frozen=True)
class PoolConfig:
    """
    Candidate pool per cluster.

    - m_center: canzoni più vicine al centroide del cluster (rappresentative)
    - m_diverse: canzoni più lontane dal centroide (diversità)
    - m_random: campione casuale dal rimanente
    - refresh_every: ogni quanti feedback ricostruire i pool
    - epsilon_out_of_pool: % volte in cui si esce dal pool (ma sempre nel cluster) per esplorare
    """
    m_center: int = 200
    m_diverse: int = 200
    m_random: int = 100
    refresh_every: int = 25
    epsilon_out_of_pool: float = 0.15
