from features.song_representation import vectorize_songs_batch
from evaluation.metrics import get_similarity, WEIGHTS
import pandas as pd
import numpy as np

N_SONGS = 10

TARGET_SIM = 0.50
SIM_SPREAD = 0.25
GENRE_WEIGHT = 0.06

DISLIKE_PENALTY = 0.25
PROFILE_BALANCE = 0.7   # how evenly we sample from taste profiles

def generate_fallback_songs(
    df: pd.DataFrame,
    user_profile,
    seen_track_ids: set,
    n_songs: int = N_SONGS
) -> pd.DataFrame:
    """
    Multi-profile fallback recommender.

    Strategy:
    - 80% from strongest / active profiles
    - 20% from weakest / inactive profiles
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

    active_profile_vectors = [
        p.vector * WEIGHTS for p in active_profiles
    ]
    active_profile_norms = [
        np.linalg.norm(v) for v in active_profile_vectors
    ]

    inactive_profiles = [tp for tp in taste_profiles if tp not in active_profiles]

    inactive_profile_vectors = [
        p.vector * WEIGHTS for p in inactive_profiles
    ]
    inactive_profile_norms = [
        np.linalg.norm(v) for v in inactive_profile_vectors
    ]
    scored = []

    # ---------- Phase 4: scoring ----------
    for row in candidates.itertuples(index=False):
        song_vec = row.vector * WEIGHTS
        song_norm = np.linalg.norm(song_vec)
        if song_norm == 0:
            continue

        # ---- find closest taste profile ----
        active_sims = [
            get_similarity(song_vec, pv, song_norm, pn)
            for pv, pn in zip(active_profile_vectors, active_profile_norms)
        ]

        inactive_sims = [
            get_similarity(song_vec, pv, song_norm, pn) 
            for pv, pn in zip(inactive_profile_vectors, inactive_profile_norms)
        ]

        best_active_idx = int(np.argmax(active_sims))
        best_active_sim = active_sims[best_active_idx]
        active_profile = active_profiles[best_active_idx]

        best_inactive_idx = int(np.argmax(inactive_sims))
        best_inactive_sim = inactive_sims[best_inactive_idx]
        inactive_profile = inactive_profiles[best_inactive_idx]

        # ---- exploration band score ----
        active_exploration_score = -abs(best_active_sim - TARGET_SIM) / SIM_SPREAD
        inactive_exploration_score = -abs(best_inactive_sim - TARGET_SIM) / SIM_SPREAD

        active_score = active_exploration_score
        inactive_score = inactive_exploration_score

        # ---- confidence weighting (important!) ----
        active_score *= (1.0 + PROFILE_BALANCE * active_profile.confidence)
        inactive_score *= (1.0 + PROFILE_BALANCE * inactive_profile.confidence)

        # ---- genre alignment bonus ----
        if row.track_genre in active_profile.genres:
            active_score += GENRE_WEIGHT

        if row.track_genre in inactive_profile.genres:
            inactive_score += GENRE_WEIGHT

        scored.append((active_score, inactive_score, row, active_profile.cluster_name))
    
    # ---------- Phase 5: rank ----------
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---------- Phase 6: balanced selection ----------
    selected = []
    used_artists = set()
    used_profiles = {}

    for _, _, row, cluster in scored:
        if row.artists in used_artists:
            continue

        used_profiles.setdefault(cluster, 0)

        # avoid one profile dominating
        if used_profiles[cluster] > n_songs * 0.4:
            continue

        selected.append(row)
        used_artists.add(row.artists)
        used_profiles[cluster] += 1

        if len(selected) >= n_songs * .9:
            break
    
    used_artists = set()

    scored.sort(key=lambda x: x[1], reverse=True)
    
    for _, _, row, cluster in scored:
        if row.artists in used_artists:
            continue

        used_profiles.setdefault(cluster, 0)

        # avoid one profile dominating
        if used_profiles[cluster] > n_songs * 0.4:
            continue

        selected.append(row)
        used_artists.add(row.artists)
        used_profiles[cluster] += 1

        if len(selected) >= n_songs * .1:
            break
    
    return pd.DataFrame(selected).reset_index(drop=True)
