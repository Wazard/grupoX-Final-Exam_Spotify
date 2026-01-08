"""
ranking_model.py

Non-linear, lightweight ranking model for multi-modal (multi taste-profile) recommendation.

This module is intended to replace a linear/logistic taste model:
- Keeps the same concept (one model per active taste profile + small exploration)
- Uses a non-linear, fast tree-based model from scikit-learn:
  HistGradientBoostingClassifier (captures non-linear interactions)

If a taste profile does not have enough labeled data (needs both likes and dislikes),
it automatically falls back to similarity-only ranking.

Expected df columns:
- track_id
- track_genre
- SIMILARITY_FEATURES columns

Depends on:
- evaluation.metrics: SIMILARITY_FEATURES, BANNED_GENRES, get_similarity, WEIGHTS
- recommender.helper.functions: build_training_data_for_profile
- user.profile: UserProfile / TasteProfile (TasteProfile must expose .genres and .vector)
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

from recommender.helper.functions import build_training_data_for_profile
from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES, WEIGHTS
from user.profile import UserProfile


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
N_TRAIN_AMOUNT = 50                 # retrain every N newly seen songs
EXPLORATION_EXTRA = 2               # how many songs to add from weak profiles
MIN_CANDIDATES_PER_PROFILE = 8      # skip profile if too few candidates
MIN_SAMPLES_TO_TRAIN = 40           # hard lock: likes+dislikes needed for training
DISLIKE_SAMPLE_WEIGHT = 0.35        # downweight dislikes during training

# Similarity fallback
SIM_FALLBACK_MIN_SIM = 0.20
SIM_DISLIKE_PENALTY = 0.25


# -------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------
def _ensure_str_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _compile_banned_pattern(banned: List[str]) -> Optional[str]:
    if not banned:
        return None
    escaped = [re.escape(x) for x in banned if x]
    if not escaped:
        return None
    return "|".join(escaped)


def _compute_disliked_centroid(
    df: pd.DataFrame,
    disliked_ids: List[str],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not disliked_ids:
        return None, None

    disliked_set = set(map(str, disliked_ids))
    sub = df[_ensure_str_ids(df["track_id"]).isin(disliked_set)]
    if sub.empty:
        return None, None

    X = sub[SIMILARITY_FEATURES].to_numpy(dtype=np.float32)
    centroid = np.mean(X, axis=0).astype(np.float32)
    centroid_w = centroid * WEIGHTS
    n = float(np.linalg.norm(centroid_w))
    if n <= 0:
        return None, None
    return centroid_w, n


def _similarity_rank(
    candidates: pd.DataFrame,
    profile_vector: np.ndarray,
    disliked_centroid_w: Optional[np.ndarray],
    disliked_centroid_norm: Optional[float],
    top_k: int,
) -> pd.DataFrame:
    if candidates.empty or top_k <= 0:
        return candidates.head(0)

    pv = profile_vector.astype(np.float32) * WEIGHTS
    pv_norm = float(np.linalg.norm(pv))
    if pv_norm <= 0:
        return candidates.head(0)

    X = candidates[SIMILARITY_FEATURES].to_numpy(dtype=np.float32)
    Xw = X * WEIGHTS
    Xn = np.linalg.norm(Xw, axis=1)

    sims = np.zeros(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        if Xn[i] > 0:
            sims[i] = np.dot(Xw[i], pv) / (Xn[i] * pv_norm)

    mask = sims >= SIM_FALLBACK_MIN_SIM
    if not np.any(mask):
        return candidates.head(0)

    sims = sims[mask]
    sub = candidates.loc[mask].copy()

    if disliked_centroid_w is not None and disliked_centroid_norm and disliked_centroid_norm > 0:
        rep = np.zeros(len(sub), dtype=np.float32)
        Xw_sub = Xw[mask]
        Xn_sub = Xn[mask]
        for i in range(len(sub)):
            if Xn_sub[i] > 0:
                rep[i] = np.dot(Xw_sub[i], disliked_centroid_w) / (Xn_sub[i] * disliked_centroid_norm)
        score = sims - (rep * SIM_DISLIKE_PENALTY)
    else:
        score = sims

    sub["score"] = score
    sub = sub.sort_values("score", ascending=False)
    return sub.head(top_k)


# -------------------------------------------------
# MODEL TRAINING (PER PROFILE)
# -------------------------------------------------
def train_profile_model_nonlinear(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    sample_weight = np.where(y == 1, 1.0, DISLIKE_SAMPLE_WEIGHT).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=12,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_ranking_models_if_needed(
    df: pd.DataFrame,
    user_profile: UserProfile,
    last_train_seen: int,
) -> int:
    seen_now = len(user_profile.seen_song_ids)
    if seen_now - last_train_seen < N_TRAIN_AMOUNT:
        return last_train_seen

    if df.empty or not user_profile.has_profile():
        return seen_now

    df = df.copy()
    df["track_id"] = _ensure_str_ids(df["track_id"])

    for profile in user_profile.taste_profiles:
        n_samples = int(getattr(profile, "liked_count", 0) + getattr(profile, "disliked_count", 0))
        if n_samples < MIN_SAMPLES_TO_TRAIN:
            profile.model = None
            profile.model_type = "similarity"
            profile.model_metrics = {"reason": f"locked_low_samples({n_samples})"}
            continue

        X, y = build_training_data_for_profile(
            df=df,
            profile=profile,
            liked_ids=user_profile.liked_song_ids,
            disliked_ids=user_profile.disliked_song_ids,
        )

        if X is None or y is None:
            profile.model = None
            profile.model_type = "similarity"
            profile.model_metrics = {"reason": "no_training_rows"}
            continue

        if len(np.unique(y)) < 2:
            profile.model = None
            profile.model_type = "similarity"
            profile.model_metrics = {"reason": "single_class"}
            continue

        model = train_profile_model_nonlinear(X, y)

        # quick training metrics (in-sample; replace with CV if you prefer)
        try:
            p = model.predict_proba(X)[:, 1]
            profile.model_metrics = {
                "auc": float(roc_auc_score(y, p)),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
                "n": int(len(y)),
            }
        except Exception:
            profile.model_metrics = {"n": int(len(y))}

        profile.model = model
        profile.model_type = "hgb"

    return seen_now


# -------------------------------------------------
# MODEL-BASED RANKING (MULTI-PROFILE)
# -------------------------------------------------
def generate_ranking_model_rank(
    df: pd.DataFrame,
    user_profile: UserProfile,
    seen_track_ids: Set[str],
    n_songs: int,
) -> pd.DataFrame:
    """
    Multi-modal model recommender.

    Strategy:
    - Even split across active taste profiles
    - Small exploration from weakest profiles
    - Per-profile: model rank if available, else similarity rank
    """
    if n_songs <= 0 or df.empty or not user_profile.has_profile():
        return pd.DataFrame()

    df = df.copy()
    df["track_id"] = _ensure_str_ids(df["track_id"])
    seen_track_ids = set(map(str, seen_track_ids))

    # Remove banned genres
    pattern = _compile_banned_pattern(BANNED_GENRES)
    if pattern:
        df = df[~df["track_genre"].astype(str).str.contains(pattern, case=False, na=False)]

    # Remove seen tracks (first pass)
    df = df[~df["track_id"].isin(seen_track_ids)]
    if df.empty:
        return pd.DataFrame()

    active_profiles = user_profile.get_active_profiles(min_confidence=1.0)
    if not active_profiles:
        return pd.DataFrame()

    weak_profiles = sorted(user_profile.taste_profiles, key=lambda p: float(getattr(p, "confidence", 0.0)))

    exploration_budget = min(EXPLORATION_EXTRA, n_songs)
    exploitation_budget = max(0, n_songs - exploration_budget)
    per_profile = max(1, exploitation_budget // len(active_profiles)) if exploitation_budget > 0 else 0

    disliked_centroid_w, disliked_centroid_norm = _compute_disliked_centroid(df, user_profile.disliked_song_ids)

    # ---- Exploitation ----
    exploited_parts: List[pd.DataFrame] = []
    for profile in active_profiles:
        genres = getattr(profile, "genres", None)
        if not genres:
            continue

        candidates = df[df["track_genre"].isin(genres)]
        if len(candidates) < MIN_CANDIDATES_PER_PROFILE:
            continue

        model = getattr(profile, "model", None)
        if model is not None and getattr(profile, "model_type", "") == "hgb":
            X = candidates[SIMILARITY_FEATURES].to_numpy(dtype=np.float32)
            probs = model.predict_proba(X)[:, 1].astype(np.float32)

            part = (
                candidates
                .assign(score=probs)
                .sort_values("score", ascending=False)
                .head(per_profile)
            )
        else:
            pv = getattr(profile, "vector", None)
            if pv is None:
                continue
            part = _similarity_rank(
                candidates=candidates,
                profile_vector=np.asarray(pv, dtype=np.float32),
                disliked_centroid_w=disliked_centroid_w,
                disliked_centroid_norm=disliked_centroid_norm,
                top_k=per_profile,
            )

        if not part.empty:
            exploited_parts.append(part)

    exploited = pd.concat(exploited_parts, axis=0) if exploited_parts else pd.DataFrame()

    # Adjust exploration based on remaining slots
    remaining_after_exploit = max(0, n_songs - len(exploited))
    exploration_budget = min(exploration_budget, remaining_after_exploit)

    # ---- Exploration from weak profiles ----
    exploration_parts: List[pd.DataFrame] = []
    for profile in weak_profiles:
        if exploration_budget <= 0:
            break
        if profile in active_profiles:
            continue

        genres = getattr(profile, "genres", None)
        if not genres:
            continue

        candidates = df[df["track_genre"].isin(genres)]
        if candidates.empty:
            continue

        take = min(exploration_budget, len(candidates))
        exploration_parts.append(candidates.sample(take, random_state=42))
        exploration_budget -= take

    final = pd.concat([exploited] + exploration_parts, axis=0) if (not exploited.empty or exploration_parts) else pd.DataFrame()

    # Defensive: unseen and unique (second pass)
    final = (
        final
        .drop_duplicates("track_id")
        .loc[~final["track_id"].isin(seen_track_ids)]
        .head(n_songs)
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # Fallback: never return empty / underfilled
    # -------------------------------------------------
    # If we couldn't score against any profile (e.g., missing models, no genre overlap,
    # all candidates filtered), fill remaining slots using similarity to a global
    # centroid of active profiles.
    if final.empty or len(final) < n_songs:
        remaining = n_songs - len(final)

        # Use active profile vectors if available, otherwise any taste profile.
        pool_profiles = active_profiles if active_profiles else user_profile.taste_profiles

        # Compute a confidence-weighted centroid (robust default).
        centroid = None
        weight_sum = 0.0
        for p in pool_profiles:
            v = getattr(p, "vector", None)
            if v is None:
                continue
            w = float(getattr(p, "confidence", 1.0))
            if w <= 0:
                w = 1.0
            v = np.asarray(v, dtype=np.float32)
            if centroid is None:
                centroid = w * v
            else:
                centroid = centroid + w * v
            weight_sum += w

        if centroid is not None and weight_sum > 0:
            centroid = centroid / weight_sum

            # Avoid selecting items already in final
            already = set(final["track_id"].astype(str)) if not final.empty else set()
            fill_candidates = df.loc[~df["track_id"].astype(str).isin(already)].copy()

            filler = _similarity_rank(
                fill_candidates,
                user_vec=centroid,
                top_k=remaining,
            )

            if not filler.empty:
                final = pd.concat([final, filler], axis=0)
                final = (
                    final
                    .drop_duplicates("track_id")
                    .head(n_songs)
                    .reset_index(drop=True)
                )

    # Last resort: if still empty, return a random sample (better than returning nothing)
    if final.empty:
        take = min(n_songs, len(df))
        if take <= 0:
            return pd.DataFrame()
        final = df.sample(take, random_state=42).reset_index(drop=True)

    return final
