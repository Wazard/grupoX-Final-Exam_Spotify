from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
from user.profile import UserProfile
import pandas as pd
import numpy as np

N_SONGS = 10

DISLIKE_PENALTY = 0.30
MIN_SIM_PROFILE = 0.40   # stricter than fallback


def generate_ranking(
    df: pd.DataFrame,
    user_profile: UserProfile,
    n_songs: int = N_SONGS
) -> pd.DataFrame:
    """
    Multi-profile exploitative ranking recommender.

    Strategy:
    - Score each song against its closest taste profile
    - Strong alignment required
    - Confidence-weighted exploitation
    """

    taste_profiles = user_profile.taste_profiles
    seen_track_ids = user_profile.seen_song_ids
    disliked_track_ids = user_profile.disliked_song_ids

    # ---------- Phase 1: candidate pool ----------
    candidates = df.dropna()
    candidates = candidates[~candidates["track_id"].isin(seen_track_ids)]

    if candidates.empty or not taste_profiles:
        return pd.DataFrame()

    # ---------- Phase 2: precompute vectors ----------
    if "vector" not in candidates.columns:
        candidates = candidates.copy()
        candidates["vector"] = list(vectorize_songs_batch(candidates))

    # ---------- Phase 3: prepare profiles ----------
    active_profiles = [
        p for p in taste_profiles
        if p.confidence > 0 and np.linalg.norm(p.vector) > 0
    ]

    if not active_profiles:
        return pd.DataFrame()

    profile_vectors = [p.vector * WEIGHTS for p in active_profiles]
    profile_norms = [np.linalg.norm(v) for v in profile_vectors]
    profile_conf = [p.confidence for p in active_profiles]

    # ---------- Phase 4: disliked centroid (global soft penalty) ----------
    disliked_df = df[df["track_id"].isin(disliked_track_ids)]
    if not disliked_df.empty:
        disliked_vectors = vectorize_songs_batch(disliked_df)
        disliked_centroid = np.mean(np.array(disliked_vectors), axis=0) * WEIGHTS
        disliked_norm = np.linalg.norm(disliked_centroid)
    else:
        disliked_centroid = None
        disliked_norm = None

    scored = []

    # ---------- Phase 5: scoring ----------
    for row in candidates.itertuples(index=False):
        song_vec = row.vector * WEIGHTS
        song_norm = np.linalg.norm(song_vec)
        if song_norm == 0:
            continue

        # ---- find best matching taste profile ----
        sims = [
            get_similarity(song_vec, pv, song_norm, pn)
            for pv, pn in zip(profile_vectors, profile_norms)
        ]

        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        # Hard reject weak matches
        if best_sim < MIN_SIM_PROFILE:
            continue

        score = best_sim * (1.0 + np.log1p(profile_conf[best_idx]))

        # ---- disliked penalty (global) ----
        if disliked_centroid is not None and disliked_norm > 0:
            sim_disliked = get_similarity(
                song_vec, disliked_centroid, song_norm, disliked_norm
            )
            score -= sim_disliked * DISLIKE_PENALTY

        scored.append((score, row, best_idx))

    if not scored:
        return pd.DataFrame()

    # ---------- Phase 6: rank ----------
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---------- Phase 7: balance across profiles ----------
    selected = []
    used_artists = set()
    profile_counts = {}

    for score, row, profile_idx in scored:
        if row.artists in used_artists:
            continue

        profile_counts.setdefault(profile_idx, 0)

        # prevent single taste domination
        if profile_counts[profile_idx] > n_songs * 0.6:
            continue

        selected.append(row)
        used_artists.add(row.artists)
        profile_counts[profile_idx] += 1

        if len(selected) == n_songs:
            break

    return pd.DataFrame(selected).reset_index(drop=True)
