from recommender.helper.functions import build_training_data_for_profile, is_better_model
from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from user.profile import UserProfile, TasteProfile
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import numpy as np


MODEL_DIR = Path("models")

# ============================================================
# CONFIG
# ============================================================

N_TRAIN_AMOUNT = 50

MIN_PROFILE_SAMPLES = 60        # HARD LOCK: no model below this
STRONG_CONFIDENCE = 1.5

STRONG_PROFILE_RATIO = 0.75     # % of recommendations
WEAK_PROFILE_RATIO = 0.25

MIN_STRONG_SLOTS = 1
EXPLORATION_RANDOM_STATE = 42

# ============================================================
# MODEL TRAINING
# ============================================================

def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute neg/pos ratio for imbalanced binary labels (LightGBM native API)."""
    y = np.asarray(y)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def _train_profile_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    **kwargs
):
    """Train a LightGBM binary model.

    Notes
    - Uses LightGBM native API (lgb.train).
    - `class_weight` is NOT supported here; use `scale_pos_weight` or per-row weights instead.
    """

    # Default parameters (safe for small, noisy per-profile datasets)
    default_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,   # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "random_state": 42,
        "n_jobs": -1,
        "metric": "binary_logloss",
        "force_col_wise": True,  # removes the col-wise auto-test overhead log
        "verbosity": -1,
    }

    # Update with user params if provided
    if params:
        default_params.update(params)

    # If caller didn't provide a validation set, create one
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y

    # Handle class imbalance (native API way)
    # Only set if caller didn't already provide one.
    if "scale_pos_weight" not in default_params:
        default_params["scale_pos_weight"] = _compute_scale_pos_weight(y_train)

    # Create datasets for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        default_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=callbacks,
        **kwargs,
    )

    return model


def train_profile_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
):
    """Train with stratified cross-validation.

    - Trains each fold using the fold's validation split (no extra internal split).
    - Reports AUC and LogLoss per fold.
    """

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores = []
    logloss_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = _train_profile_model(X_train, y_train, X_val=X_val, y_val=y_val)
        y_pred = model.predict(X_val)

        # Metrics are undefined if validation has a single class
        if len(np.unique(y_val)) < 2:
            print(
                f"Fold {fold}: only one class in y_val "
                f"(labels={np.unique(y_val)}), skipping metrics."
            )
        else:
            fold_auc = roc_auc_score(y_val, y_pred)
            fold_ll = log_loss(y_val, y_pred, labels=[0, 1])

            auc_scores.append(fold_auc)
            logloss_scores.append(fold_ll)

            print(
                f"Fold {fold}: AUC = {fold_auc:.4f} | "
                f"LogLoss = {fold_ll:.4f}"
            )

        models.append(model)

        print(f"Fold {fold}: AUC = {fold_auc:.4f} | LogLoss = {fold_ll:.4f}")

    # Train final model on all data (with an internal holdout for early stopping)
    final_model = _train_profile_model(X, y)

    cv_results = {
        "mean_auc": float(np.mean(auc_scores)),
        "std_auc": float(np.std(auc_scores)),
        "fold_auc_scores": auc_scores,
        "mean_logloss": float(np.mean(logloss_scores)),
        "std_logloss": float(np.std(logloss_scores)),
        "fold_logloss_scores": logloss_scores,
        "models": models,
        "final_model": final_model,
    }

    print(f"\nCV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"CV LogLoss: {cv_results['mean_logloss']:.4f} ± {cv_results['std_logloss']:.4f}")

    return final_model, cv_results


def log_model_performance(profile: TasteProfile, X, y):
    """Quick diagnostic on the provided dataset (often training data)."""
    probs = profile.model.predict(X)

    auc = roc_auc_score(y, probs)
    loss = log_loss(y, probs)

    print(
        f"[LGBM] profile={profile.cluster_name} | "
        f"samples={len(y)} | "
        f"AUC={auc:.3f} | "
        f"LogLoss={loss:.3f} | "
        f"confidence={profile.confidence:.2f}"
    )

# ============================================================
# TRAIN MODELS IF NEEDED
# ============================================================

def train_lgbms_if_needed(
    df: pd.DataFrame,
    user_profile: UserProfile,
    last_train_seen: int,
):
    seen_now = len(user_profile.seen_song_ids)

    if seen_now - last_train_seen < N_TRAIN_AMOUNT:
        return last_train_seen

    for profile in user_profile.taste_profiles:
        total_samples = profile.liked_count + profile.disliked_count

        # HARD LOCK
        if total_samples < MIN_PROFILE_SAMPLES:
            profile.model = None
            continue

        # Load existing model (if any)
        load_existing_model(profile)

        X, y = build_training_data_for_profile(
            df,
            profile,
            user_profile.liked_song_ids,
            user_profile.disliked_song_ids,
        )

        if X is None:
            profile.model = None
            continue

        new_model, cv_res = train_profile_model_cv(X, y)

        new_logloss = cv_res.get("mean_logloss")
        new_auc = cv_res.get("mean_auc")

        if is_better_model(
            getattr(profile, "best_logloss", None),
            getattr(profile, "best_auc", None),
            new_logloss,
            new_auc,
        ):
            profile.model = new_model
            profile.best_logloss = new_logloss
            profile.best_auc = new_auc

            new_model.save_model(str(_model_path(profile)))

            print(
                f"[LGBM] Updated model for {profile.cluster_name} | "
                f"logloss={new_logloss:.4f} | auc={new_auc:.4f}"
            )
        else:
            print(
                f"[LGBM] Kept existing model for {profile.cluster_name}"
            )

        log_model_performance(profile, X, y)

    return seen_now


# ============================================================
# MODEL-BASED RANKING (MULTI-MODAL)
# ============================================================

def generate_lgbm_rank(
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

    # Remove seen tracks
    df = df[~df["track_id"].isin(seen_track_ids)]

    if df.empty:
        return pd.DataFrame()

    # -------------------------------
    # PROFILE SPLIT
    # -------------------------------
    strong_profiles = [
        p for p in user_profile.taste_profiles
        if p.confidence >= STRONG_CONFIDENCE
    ]

    weak_profiles = [
        p for p in user_profile.taste_profiles
        if p.confidence < STRONG_CONFIDENCE
    ]

    if not strong_profiles:
        strong_profiles = weak_profiles[:1]

    strong_slots = max(
        MIN_STRONG_SLOTS,
        int(n_songs * STRONG_PROFILE_RATIO)
    )
    weak_slots = n_songs - strong_slots

    per_strong = max(1, strong_slots // len(strong_profiles))
    per_weak = max(1, weak_slots // max(1, len(weak_profiles)))

    results = []

    # -------------------------------
    # STRONG PROFILES (EXPLOIT)
    # -------------------------------
    for profile in strong_profiles:
        candidates = df[df["track_genre"].isin(profile.genres)]
        if candidates.empty:
            continue

        if profile.model is None:
            ranked = candidates.sample(
                min(per_strong, len(candidates)),
                random_state=EXPLORATION_RANDOM_STATE
            )
        else:
            X = candidates[SIMILARITY_FEATURES].values.astype(np.float32)
            probs = profile.model.predict(X)
            ranked = (
                candidates
                .assign(score=probs)
                .sort_values("score", ascending=False)
                .head(per_strong)
            )

        results.append(ranked)

    # -------------------------------
    # WEAK PROFILES (CONTROLLED EXPLORE)
    # -------------------------------
    weak_used = 0

    for profile in sorted(weak_profiles, key=lambda p: p.confidence):
        if weak_used >= weak_slots:
            break

        candidates = df[df["track_genre"].isin(profile.genres)]
        if candidates.empty:
            continue

        sampled = candidates.sample(
            min(per_weak, len(candidates)),
            random_state=EXPLORATION_RANDOM_STATE
        )

        results.append(sampled)
        weak_used += len(sampled)

    # -------------------------------
    # FINAL MERGE
    # -------------------------------
    final = pd.concat(results, axis=0)
    final = final.drop_duplicates("track_id").head(n_songs)

    return final.reset_index(drop=True)


def _model_path(profile: TasteProfile) -> Path:
    return MODEL_DIR / f"{profile.cluster_name}.txt"


def load_existing_model(profile: TasteProfile):
    path = _model_path(profile)
    if path.exists():
        profile.model = lgb.Booster(model_file=str(path))
