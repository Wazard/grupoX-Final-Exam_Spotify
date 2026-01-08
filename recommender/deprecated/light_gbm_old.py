from recommender.helper.functions import build_training_data_for_profile
from evaluation.metrics import SIMILARITY_FEATURES, BANNED_GENRES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from user.profile import UserProfile, TasteProfile
import lightgbm as lgb
import pandas as pd
import numpy as np


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

def _train_profile_model(
    X: np.ndarray, 
    y: np.ndarray,
    params: dict = None,
    **kwargs
):
    """Train with customizable parameters."""
    
    # Default parameters
    default_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'metric': 'binary_logloss'
    }
    
    # Update with user params if provided
    if params:
        default_params.update(params)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets for LightGBM (faster)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train with callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=100)
    ]
    
    # Train
    model = lgb.train(
        default_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=callbacks,
        **kwargs
    )
    
    return model

def train_profile_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5
):
    """Train with cross-validation."""
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = _train_profile_model(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        fold_accuracy = roc_auc_score(y_val, y_pred)
        scores.append(fold_accuracy)
        models.append(model)
        
        print(f"Fold {fold+1}: Accuracy = {fold_accuracy:.4f}")
    
    # Train final model on all data
    final_model = _train_profile_model(X, y)
    
    cv_results = {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'fold_scores': scores,
        'models': models,
        'final_model': final_model
    }
    
    print(f"\nCV Results: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return final_model, cv_results


def log_model_performance(profile: TasteProfile, X, y):
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

        X, y = build_training_data_for_profile(
            df,
            profile,
            user_profile.liked_song_ids,
            user_profile.disliked_song_ids,
        )

        if X is None:
            profile.model = None
            continue

        profile.model, cv_res = train_profile_model_cv(X, y)
        print(f"CV: {cv_res}")
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
            ranked = candidates.assign(score=probs).sort_values("score", ascending=False).head(per_strong)

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
