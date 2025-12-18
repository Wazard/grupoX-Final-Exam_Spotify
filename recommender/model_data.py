import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES


# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------
def build_training_data(
    df: pd.DataFrame,
    liked_ids: list[str],
    disliked_ids: list[str],
    feature_cols: list[str] = SIMILARITY_FEATURES,
):
    """
    Build (X, y) training data from user feedback.

    Likes → label 1
    Dislikes → label 0
    """

    # Ensure consistent typing
    liked_ids = set(map(str, liked_ids))
    disliked_ids = set(map(str, disliked_ids))

    liked = df[df["track_id"].isin(liked_ids)].copy()
    disliked = df[df["track_id"].isin(disliked_ids)].copy()

    if liked.empty and disliked.empty:
        raise ValueError("No training data available")

    liked["label"] = 1
    disliked["label"] = 0

    data = pd.concat([liked, disliked], axis=0)

    X = data[feature_cols].values.astype(np.float32)
    y = data["label"].values.astype(np.int32)

    return X, y


# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
def train_like_model(X: np.ndarray, y: np.ndarray):
    """
    Train a logistic regression model to predict likeability.

    Dislikes are downweighted to avoid 'safe content' collapse.
    """

    # Likes are strong signals, dislikes are weaker
    sample_weight = np.where(y == 1, 1.0, 0.3)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
    )

    model.fit(X, y, sample_weight=sample_weight)
    return model


# -------------------------------------------------
# MODEL-BASED RANKING
# -------------------------------------------------
def generate_model_rank(
    df: pd.DataFrame,
    model,
    seen_track_ids: set[str],
    n_songs: int,
    feature_cols: list[str] = SIMILARITY_FEATURES,
):
    """
    Rank songs using the trained model.

    Steps:
    1. Remove non-music / functional audio (sleep, white noise, etc.)
    2. Remove already seen tracks
    3. Score remaining tracks with the model
    4. Return top N
    """

    if df.empty:
        return pd.DataFrame()

    # --- Phase 0: remove banned genres (CRITICAL) ---
    pattern = "|".join(BANNED_GENRES)
    candidates = df[
        ~df["track_genre"].str.contains(pattern, case=False, na=False)
    ].copy()

    # --- Phase 1: remove already seen tracks ---
    seen_track_ids = set(map(str, seen_track_ids))
    candidates["track_id"] = candidates["track_id"].astype(str)

    candidates = candidates[~candidates["track_id"].isin(seen_track_ids)]

    if candidates.empty:
        return pd.DataFrame()

    # --- Phase 2: model inference ---
    X = candidates[feature_cols].values.astype(np.float32)

    probs = model.predict_proba(X)[:, 1]
    candidates["score"] = probs

    # --- Phase 3: ranking ---
    candidates = candidates.sort_values("score", ascending=False)

    return candidates.head(n_songs)
