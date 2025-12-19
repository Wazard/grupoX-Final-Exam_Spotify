from evaluation.metrics import SIMILARITY_FEATURES
from user.profile import TasteProfile
import pandas as pd
import numpy as np



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

    liked["target"] = 1
    disliked["target"] = 0

    data = pd.concat([liked, disliked], axis=0)

    X = data[SIMILARITY_FEATURES].values.astype(np.float32)
    y = data["target"].values.astype(np.int32)

    return X, y


def is_better_model(
    old_logloss: float | None,
    old_auc: float | None,
    new_logloss: float | None,
    new_auc: float | None,
) -> bool:
    if old_logloss is None:
        return True
    if new_logloss is None:
        return False

    # Primary metric
    if new_logloss < old_logloss - 1e-4:
        return True

    # Secondary metric
    if abs(new_logloss - old_logloss) < 1e-4:
        return (new_auc or 0.0) > (old_auc or 0.0)

    return False
