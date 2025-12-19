from evaluation.metrics import GENRE_CLUSTERS
import pandas as pd
import numpy as np

COLD_START_CLUSTERS = 2
POPULARITY_THRESHOLD = 30


def generate_cold_start_songs(
    df: pd.DataFrame,
    n_songs: int
) -> pd.DataFrame:
    """
    Multi-cluster cold start recommender.

    Strategy:
    - Pick K genre clusters
    - Sample evenly from each
    - Enforce artist uniqueness
    - Favor reasonably popular tracks
    """

    df = df.dropna().copy()
    df = df[df["popularity"] >= POPULARITY_THRESHOLD]

    # ---------- Pick clusters ----------
    cluster_names = list(GENRE_CLUSTERS.keys())
    chosen_clusters = np.random.choice(
        cluster_names,
        size=min(COLD_START_CLUSTERS, len(cluster_names)),
        replace=False
    )

    per_cluster = max(1, n_songs // len(chosen_clusters))
    selected_rows = []
    used_artists = set()

    # ---------- Sample from each cluster ----------
    for cluster in chosen_clusters:
        genres = set(GENRE_CLUSTERS[cluster])

        cluster_df = df[df["track_genre"].isin(genres)]
        if cluster_df.empty:
            continue

        # Shuffle to avoid popularity-only bias
        cluster_df = cluster_df.sample(frac=1)

        for _, row in cluster_df.iterrows():
            if row["artists"] in used_artists:
                continue

            selected_rows.append(row)
            used_artists.add(row["artists"])

            if len(selected_rows) % per_cluster == 0:
                break

    # ---------- Fallback if not enough ----------
    if len(selected_rows) < n_songs:
        remaining = df[~df["artists"].isin(used_artists)]
        remaining = remaining.sample(frac=1)

        for _, row in remaining.iterrows():
            selected_rows.append(row)
            if len(selected_rows) == n_songs:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)
