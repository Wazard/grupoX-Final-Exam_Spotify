from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
import pandas as pd
import numpy as np


N_SONGS = 10
POPULARITY_THRESHOLD = 75

MAX_SIM_DISLIKED = 0.85
MIN_SIM_LIKED = 0.65


def generate_fallback_songs(
    df: pd.DataFrame,
    liked_vectors: list[list[float]],
    disliked_vectors: list[list[float]],
    seen_track_ids: set,
    n_songs: int = N_SONGS
) -> pd.DataFrame:
    """
    Optimized fallback recommender.

    Optimizations applied:
    - Precomputed song vectors (no repeated vectorization)
    - Precomputed norms for similarity (no repeated norm computation)
    - Replaced iterrows() with itertuples()
    """

    # --- Phase 1: candidate pool pruning ---
    candidates = df.dropna()
    candidates = candidates[~candidates["track_id"].isin(seen_track_ids)]
    candidates = candidates[candidates["popularity"] < POPULARITY_THRESHOLD]

    # --- Phase 2: compute genre rarity weights ---
    genre_counts = candidates["track_genre"].value_counts()
    genre_weights = {
        genre: 1 / np.sqrt(count)
        for genre, count in genre_counts.items()
    }

    # --- OPTIMIZATION 1: precompute vectors once ---
    # Avoids calling vectorize_song inside the loop
    if "vector" not in candidates.columns:
        candidates = candidates.copy()
        candidates["vector"] = list(vectorize_songs_batch(candidates))

    # --- OPTIMIZATION 2: convert liked/disliked vectors to NumPy ---
    np_liked_vectors = np.array(liked_vectors)
    np_disliked_vectors = np.array(disliked_vectors)

    weighted_liked_norms = np.linalg.norm(np_liked_vectors * WEIGHTS, axis=1)          #  calculating weights out of get_similarity_weighted 
    weighted_disliked_norms = np.linalg.norm(np_disliked_vectors * WEIGHTS, axis=1)    #          as it's faster doing it all at once

    scored_candidates = []

    # --- OPTIMIZATION 3: use itertuples (much faster than iterrows) ---
    for row in candidates.itertuples(index=False):

        song_vec = row.vector * WEIGHTS         # Again, weighting outside of get_similarity
        song_norm = np.linalg.norm(song_vec)

        max_sim_disliked = 0.0
        for dv, dv_norm in zip(np_disliked_vectors, weighted_disliked_norms):
            sim = get_similarity(song_vec, dv, song_norm, dv_norm)
            if sim > MAX_SIM_DISLIKED:
                max_sim_disliked = sim
                break

        if max_sim_disliked > MAX_SIM_DISLIKED:
            continue  # skip bad region entirely

        max_sim_liked = 0.0
        for lv, lv_norm in zip(np_liked_vectors, weighted_liked_norms):
            sim = get_similarity(song_vec, lv, song_norm, lv_norm)
            if sim > max_sim_liked:
                max_sim_liked = sim

        # --- scoring ---
        score = 0.0

        if max_sim_liked > MIN_SIM_LIKED:
            score += max_sim_liked

        score += genre_weights.get(row.track_genre, 0.0)
        score -= max_sim_disliked * 0.5

        scored_candidates.append((score, row))

    # --- Phase 3: sort by score ---
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Phase 4: diversity-aware selection ---
    selected_rows = []
    used_artists = set()
    used_genres = set()

    for score, row in scored_candidates:
        if row.artists in used_artists:
            continue
        if row.track_genre in used_genres:
            continue

        selected_rows.append(row)
        used_artists.add(row.artists)
        used_genres.add(row.track_genre)

        if len(selected_rows) == n_songs:
            break

    return pd.DataFrame(selected_rows)
