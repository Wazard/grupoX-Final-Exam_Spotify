from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES
from sklearn.linear_model import LogisticRegression
from user.profile import TasteProfile, UserProfile
import pandas as pd
import numpy as np


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
N_TRAIN_AMOUNT = 50
DISLIKE_WEIGHT = 0.35
EXPLORATION_EXTRA = 2   # extra songs from low-confidence profiles


# -------------------------------------------------
# TRAINING DATA (PER PROFILE)
# -------------------------------------------------
def build_training_data_for_profile(
    df: pd.DataFrame,
    profile: TasteProfile,
    liked_ids: list[str],
    disliked_ids: list[str],
):
    """
    Build training data ONLY from genres belonging
    to the given taste profile.
    """

    liked_ids = set(map(str, liked_ids))
    disliked_ids = set(map(str, disliked_ids))

    mask = df["track_genre"].isin(profile.genres)
    subset = df[mask]

    liked = subset[subset["track_id"].isin(liked_ids)].copy()
    disliked = subset[subset["track_id"].isin(disliked_ids)].copy()

    if liked.empty and disliked.empty:
        return None, None

    liked["label"] = 1
    disliked["label"] = 0

    data = pd.concat([liked, disliked], axis=0)

    X = data[SIMILARITY_FEATURES].values.astype(np.float32)
    y = data["label"].values.astype(np.int32)

    return X, y


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
def generate_model_rank(
    df: pd.DataFrame,
    user_profile: UserProfile,
    seen_track_ids: set[str],
    n_songs: int,
):
    """
    Multi-modal model recommender.

    Strategy:
    - One model per active taste profile
    - Split recommendations evenly
    - Add small exploration from weakest profiles
    """

    if df.empty or not user_profile.has_profile():
        return pd.DataFrame()
    
    df = df.copy()
    # Remove banned genres
    pattern = "|".join(BANNED_GENRES)
    df = df[~df["track_genre"].str.contains(pattern, case=False, na=False)]

    # Remove seen tracks
    df = df[~df["track_id"].isin(seen_track_ids)]

    if df.empty:
        return pd.DataFrame()

    # -----------------------------------------
    # ACTIVE & WEAK PROFILES
    # -----------------------------------------
    active_profiles = user_profile.get_active_profiles(min_confidence=1.0)

    if not active_profiles:
        return pd.DataFrame()

    inactive_profiles = sorted(
        user_profile.taste_profiles,
        key=lambda p: p.confidence
    )

    per_profile = max(1, n_songs // len(active_profiles))
    results = []

    # -----------------------------------------
    # EXPLOITATION: ACTIVE PROFILES
    # -----------------------------------------
    for profile in active_profiles:
        candidates = df[df["track_genre"].isin(profile.genres)]
        if candidates.empty:
            continue

        X = candidates[SIMILARITY_FEATURES].values.astype(np.float32)

        model = getattr(profile, "model", None)
        if model is None:
            continue

        probs = model.predict_proba(X)[:, 1]
        candidates = candidates.assign(score=probs)
        candidates = candidates.sort_values("score", ascending=False)

        results.append(candidates.head(per_profile))

    # -----------------------------------------
    # EXPLORATION: WEAK PROFILES
    # -----------------------------------------
    extra = []

    for profile in inactive_profiles:
        if profile.confidence > 0:
            continue

        candidates = df[df["track_genre"].isin(profile.genres)]
        if candidates.empty:
            continue

        extra.append(
            candidates.sample(
                min(EXPLORATION_EXTRA, len(candidates)),
                random_state=42
            )
        )

        if sum(len(x) for x in extra) >= EXPLORATION_EXTRA:
            break

    # -----------------------------------------
    # FINAL MERGE
    # -----------------------------------------
    final = pd.concat(results + extra, axis=0)

    final = final.drop_duplicates("track_id")
    final = final.head(n_songs)

    return final.reset_index(drop=True)


# -------------------------------------------------
# TRAIN / UPDATE ALL PROFILE MODELS
# -------------------------------------------------
def train_models_if_needed(
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

    for profile in user_profile.taste_profiles:
        X, y = build_training_data_for_profile(
            df,
            profile,
            user_profile.liked_song_ids,
            user_profile.disliked_song_ids,
        )

        if X is None:
            continue

        profile.model = train_profile_model(X, y)

    return seen_now
