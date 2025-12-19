from recommender.helper.functions import build_training_data_for_profile
from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES
from sklearn.linear_model import LogisticRegression
from user.profile import UserProfile
import pandas as pd
import numpy as np


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
N_TRAIN_AMOUNT = 50
DISLIKE_WEIGHT = 0.35
EXPLORATION_EXTRA = 2   # extra songs from low-confidence profiles


# -------------------------------------------------
# MODEL TRAINING (PER PROFILE)
# -------------------------------------------------
def train_profile_model(X: np.ndarray, y: np.ndarray):
    """
    Logistic regression tuned to avoid
    'safe content collapse'.
    """

    # Downweight dislikes
    sample_weight = np.where(y == 1, 1.0, DISLIKE_WEIGHT)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
    )

    model.fit(X, y, sample_weight=sample_weight)
    return model


# -------------------------------------------------
# MODEL-BASED RANKING (MULTI-PROFILE)
# -------------------------------------------------
def generate_linear_model_rank(
    df: pd.DataFrame,
    user_profile: UserProfile,
    seen_track_ids: set[str],
    n_songs: int,
):
    if df.empty or not user_profile.has_profile():
        return pd.DataFrame()

    df = df.copy()

    # Remove banned genres
    pattern = "|".join(BANNED_GENRES)
    df = df[~df["track_genre"].str.contains(pattern, case=False, na=False)]

    # Remove seen tracks (FIRST pass)
    df = df[~df["track_id"].isin(seen_track_ids)]
    if df.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # PROFILE SPLITS
    # -----------------------------------------
    active_profiles = user_profile.get_active_profiles(min_confidence=1.0)
    if not active_profiles:
        return pd.DataFrame()

    weak_profiles = sorted(
        user_profile.taste_profiles,
        key=lambda p: p.confidence
    )

    exploitation_budget = n_songs - EXPLORATION_EXTRA
    per_profile = max(1, exploitation_budget // len(active_profiles))

    results = []

    # -----------------------------------------
    # EXPLOITATION (ACTIVE PROFILES)
    # -----------------------------------------
    for profile in active_profiles:
        model = getattr(profile, "model", None)
        if model is None:
            continue

        candidates = df[df["track_genre"].isin(profile.genres)]
        if len(candidates) < 5:
            continue

        X = candidates[SIMILARITY_FEATURES].values.astype(np.float32)
        probs = model.predict_proba(X)[:, 1]

        ranked = (
            candidates
            .assign(score=probs)
            .sort_values("score", ascending=False)
            .head(per_profile)
        )

        results.append(ranked)

    exploited = pd.concat(results, axis=0) if results else pd.DataFrame()

    # -----------------------------------------
    # EXPLORATION (WEAK PROFILES)
    # -----------------------------------------
    remaining = n_songs - len(exploited)
    exploration_budget = min(EXPLORATION_EXTRA, remaining)

    exploration = []
    for profile in weak_profiles:
        if profile in active_profiles:
            continue
        if exploration_budget <= 0:
            break

        candidates = df[df["track_genre"].isin(profile.genres)]
        if candidates.empty:
            continue

        take = min(exploration_budget, len(candidates))
        exploration.append(candidates.sample(take))
        exploration_budget -= take

    # -----------------------------------------
    # FINAL MERGE (SECOND SEEN FILTER IS CRUCIAL)
    # -----------------------------------------
    final = pd.concat([exploited] + exploration, axis=0)

    final = (
        final
        .drop_duplicates("track_id")
        .loc[~final["track_id"].isin(seen_track_ids)]
        .head(n_songs)
        .reset_index(drop=True)
    )

    return final

# -------------------------------------------------
# TRAIN / UPDATE ALL PROFILE MODELS
# -------------------------------------------------
def train_linear_models_if_needed(
    df: pd.DataFrame,
    user_profile: UserProfile,
    last_train_seen: int,
):
    """
    Train / refresh models only every N_TRAIN_AMOUNT interactions.
    """

    seen_now = len(user_profile.seen_song_ids)

    if seen_now - last_train_seen < N_TRAIN_AMOUNT:
        return last_train_seen

    print("training new model...")

    for profile in user_profile.taste_profiles:
        X, y = build_training_data_for_profile(
            df,
            profile,
            user_profile.liked_song_ids,
            user_profile.disliked_song_ids,
        )

        if X is None:
            continue

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Not enough info to train a classifier
            profile.model = None
            profile.model_type = "similarity"
            continue

        profile.model = train_profile_model(X, y)
        profile.model_type = "classifier"

    return seen_now
