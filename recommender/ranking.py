from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
import pandas as pd
import numpy as np

GENRE_WEIGHT = .1

def generate_ranking(df: pd.DataFrame, user_vector: np.ndarray[float], seen_track_ids: set, n_songs: int):

    # --- Phase 1: remotion seen tracks ---
    candidates = df.dropna()
    candidates = candidates[~candidates["track_id"].isin(seen_track_ids)]

    # --- Phase 2: weighting by genre ---
    genre_counts = candidates["track_genre"].value_counts()
    genre_weights = {
        genre: 1 / np.sqrt(count)
        for genre, count in genre_counts.items()
    }
    # Calculating user norms
    user_vector_weighted = user_vector * WEIGHTS
    user_vector_norm = np.linalg.norm(user_vector_weighted)

    scored_candidates = []

    # --- Phase 3: vectorize songs ---
    if "vector" not in candidates.columns:
        candidates = candidates.copy()
        candidates["vector"] = list(vectorize_songs_batch(candidates))

    # --- Phase 4: scoring ---
    for row in candidates.itertuples(index=False):

        song_vec = row.vector * WEIGHTS
        song_norm = np.linalg.norm(song_vec)
        score = get_similarity(song_vec, user_vector, song_norm, user_vector_norm)

        score += genre_weights.get(row.track_genre, 0.0) * GENRE_WEIGHT
        scored_candidates.append((score, row))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Phase 5: diversity-aware selection ---
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