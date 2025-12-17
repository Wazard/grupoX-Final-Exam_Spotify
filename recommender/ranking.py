from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
import pandas as pd
import numpy as np

GENRE_WEIGHT = .2

def generate_ranking(df: pd.DataFrame, user_vector: np.ndarray[float], liked_track_ids:np.ndarray[float], disliked_track_ids:np.ndarray[float], n_songs: int):

    # Define seen tracks
    seen_track_ids = set(liked_track_ids + disliked_track_ids)
    
    # Define user's song duration preference
    liked_durations = df[df["track_id"].isin(liked_track_ids)]["duration_s"]

    user_duration_mean = liked_durations.mean()
    user_duration_std = liked_durations.std()

    # Define user's explicit preference
    liked_explicit_ratio = (df[df["track_id"].isin(liked_track_ids)]["explicit"].mean())
    disliked_explicit_ratio = (df[df["track_id"].isin(disliked_track_ids)]["explicit"].mean())

    explicit_preference = 2
    if liked_explicit_ratio > 0.6 and disliked_explicit_ratio < 0.4:
        explicit_preference = 1
    elif liked_explicit_ratio < 0.4 and disliked_explicit_ratio > 0.6:
        explicit_preference = 0

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

        if explicit_preference == 1 and row.explicit:
            score += 0.03
        elif explicit_preference == 0 and row.explicit:
            score -= 0.03

        if not np.isnan(user_duration_std) and user_duration_std > 0:
            z = abs(row.duration_s - user_duration_mean) / user_duration_std
            duration_penalty = min(z * 0.05, 0.15)
            score -= duration_penalty

    
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