import numpy as np
import pandas as pd
from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
from user.profile import UserProfile

N_SONGS = 10

TARGET_SIM = 0.50
SIM_SPREAD = 0.25

DISLIKE_PENALTY = 0.25
PROFILE_BALANCE = 0.7   # how evenly sample from taste profiles


def generate_fallback_songs(
    df: pd.DataFrame,
    user_profile: UserProfile,
    n_songs: int = N_SONGS
) -> pd.DataFrame:
    """
    Multi-profile fallback recommender.

    Strategy:
    - Each taste profile defines an exploration band
    - Songs are scored against their closest profile
    - Exploration happens *around* tastes, not across the whole space
    """
    taste_profiles = user_profile.taste_profiles
    seen_track_ids = user_profile.seen_song_ids
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

    profile_vectors = [
        p.vector * WEIGHTS for p in active_profiles
    ]
    profile_norms = [
        np.linalg.norm(v) for v in profile_vectors
    ]

    scored = []

    # ---------- Phase 4: scoring ----------
    for row in candidates.itertuples(index=False):
        song_vec = row.vector * WEIGHTS
        song_norm = np.linalg.norm(song_vec)
        if song_norm == 0:
            continue

        # ---- find closest taste profile ----
        sims = [
            get_similarity(song_vec, pv, song_norm, pn)
            for pv, pn in zip(profile_vectors, profile_norms)
        ]

        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        profile = active_profiles[best_idx]

        # ---- exploration band score ----
        exploration_score = -abs(best_sim - TARGET_SIM) / SIM_SPREAD

        score = exploration_score

        # ---- confidence weighting (important!) ----
        score *= (1.0 + PROFILE_BALANCE * profile.confidence)

        # ---- genre alignment bonus ----
        if row.track_genre in profile.genres:
            score += 0.05

        scored.append((score, row, profile.cluster_name))

    if not scored:
        return pd.DataFrame()

    # ---------- Phase 5: rank ----------
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---------- Phase 6: balanced selection ----------
    selected = []
    used_artists = set()
    used_profiles = {}

    for score, row, cluster in scored:
        if row.artists in used_artists:
            continue

        used_profiles.setdefault(cluster, 0)

        # avoid one profile dominating
        if used_profiles[cluster] > n_songs * 0.4:
            continue

        selected.append(row)
        used_artists.add(row.artists)
        used_profiles[cluster] += 1

        if len(selected) == n_songs:
            break

    return pd.DataFrame(selected).reset_index(drop=True)
