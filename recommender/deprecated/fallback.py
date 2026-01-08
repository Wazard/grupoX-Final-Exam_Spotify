import pandas as pd
import numpy as np

from features.song_representation import vectorize_song
from evaluation.metrics import get_similarity

N_SONGS = 10
POPULARITY_THRESHOLD = 85

MAX_SIM_DISLIKED = 0.85
MIN_SIM_LIKED = 0.65


def generate_fallback_songs(
        df: pd.DataFrame, 
        liked_vectors: list[list[float]], disliked_vectors: list[list[float]], 
        seen_track_ids: set, n_songs: int = N_SONGS) -> pd.DataFrame:
    """
    Generate a fallback recommendation list when the user rejected most suggestions.

    Strategy:
    1. Filter unseen songs
    2. Lower popularity bias
    3. Remove songs too similar to disliked ones
    4. Score remaining candidates
    5. Select top N with artist & genre diversity
    """

    # ---------- Phase 1: candidate pool ----------
    candidates = df.copy()
    candidates = candidates.dropna()

    # Remove already seen songs
    candidates = candidates[~candidates["track_id"].isin(seen_track_ids)]

    # Lower popularity bias (exploration)
    candidates = candidates[candidates["popularity"] < POPULARITY_THRESHOLD]

    # ---------- Phase 2: genre rarity weights ----------
    genre_counts = candidates["track_genre"].value_counts()
    genre_weights = {
        genre: 1 / np.sqrt(count)
        for genre, count in genre_counts.items()
    }

    scored_candidates = []

    # ---------- Phase 3: scoring ----------
    for _, row in candidates.iterrows():
        song_vector = vectorize_song(row)

        # Similarity to disliked songs
        max_sim_disliked = (
            max(get_similarity(song_vector, v) for v in disliked_vectors)
            if disliked_vectors else 0.0
        )

        # Hard reject songs too close to disliked ones
        if max_sim_disliked > MAX_SIM_DISLIKED:
            continue

        # Similarity to liked songs
        max_sim_liked = (
            max(get_similarity(song_vector, v) for v in liked_vectors)
            if liked_vectors else 0.0
        )

        score = 0.0

        # Boost if similar to liked
        if max_sim_liked > MIN_SIM_LIKED:
            score += max_sim_liked

        # Genre rarity bonus
        score += genre_weights.get(row["track_genre"], 0.0)

        # Penalty for being close to disliked
        score -= max_sim_disliked * 0.5

        scored_candidates.append((score, row))

    # ---------- Phase 4: sort by score ----------
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # ---------- Phase 5: diversity-aware selection ----------
    selected_rows = []
    used_artists = set()
    used_genres = set()

    for score, row in scored_candidates:
        artist = row["artists"]
        genre = row["track_genre"]

        if artist in used_artists:
            continue

        if genre in used_genres:
            continue

        selected_rows.append(row)
        used_artists.add(artist)
        used_genres.add(genre)

        if len(selected_rows) == n_songs:
            break

    return pd.DataFrame(selected_rows)
