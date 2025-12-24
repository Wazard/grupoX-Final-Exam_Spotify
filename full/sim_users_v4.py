
"""
Simulators for benchmarking the music recommender.

Includes:
- FeatureUserSimulator: latent linear preferences over numeric features (+ optional genre bias),
  with optional hard or gradual shock.
- MacroGenreProbUserSimulator: preferences over macro-genres (arms = group_genre_to_macro),
  probabilistic likes (p_high for preferred macros, p_low otherwise), with optional shock.
- RandomUserSimulator: likes each song with p=0.5, independent.

All simulators expose:
- set_turn(t): to update internal time for shock schedules (call before rate()).
- rate(row) -> (reward:int, p_true:float)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import math


def sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)


# -----------------------
# Macro-genre grouping
# -----------------------

def _norm_genre(g: str) -> str:
    return str(g or "").strip().lower()


def group_genre_to_macro(genre: str) -> str:
    """
    Same mapping idea you used in sim_user_genre_only_v2.py.
    Keep it here so benchmarking is self-contained.
    """
    g = _norm_genre(genre)
    if g == "":
        return "unknown"

    if "metal" in g or g in {"grindcore", "hardcore"}:
        return "metal"

    if g in {"classical", "opera", "piano", "new-age"}:
        return "classical_like"
    if g in {"show-tunes", "disney"}:
        return "soundtrack_show"

    if "rock" in g or g in {"grunge", "emo", "punk", "ska", "goth", "guitar"} or "punk" in g:
        return "rock"

    if any(k in g for k in [
        "techno", "house", "edm", "dub", "dubstep", "electro", "electronic", "trance",
        "drum-and-bass", "breakbeat", "garage", "hardstyle", "club", "minimal-techno", "idm",
        "deep-house", "detroit-techno", "chicago-house", "progressive-house", "trip-hop"
    ]):
        return "electronic_dance"

    if g in {"hip-hop", "r-n-b", "soul", "funk"}:
        return "hiphop_rnb_soul"

    if g in {"jazz", "blues"}:
        return "jazz_blues"

    if g in {"pop", "synth-pop", "power-pop", "indie-pop", "pop-film", "party", "happy", "sad", "sleep", "study"}:
        return "pop_mood"

    if g in {"country", "honky-tonk", "bluegrass", "folk", "singer-songwriter", "songwriter"}:
        return "country_folk"

    if g in {"latin", "latino", "reggaeton", "salsa", "samba", "brazil", "mpb", "forro", "pagode",
             "sertanejo", "spanish", "romance"}:
        return "latin_brazil"

    if g in {"reggae", "dancehall"}:
        return "reggae_dancehall"

    if g in {"world-music", "turkish", "iranian", "indian", "malay", "german", "french", "swedish", "british",
             "cantopop", "mandopop"}:
        return "world_regional"
    if g in {"j-pop", "j-rock", "j-idol", "j-dance", "k-pop", "anime"}:
        return "east_asia_pop"

    if g in {"children", "kids", "comedy"}:
        return "kids_comedy"

    if g in {"disco"}:
        return "disco"

    if g in {"acoustic", "ambient", "chill"}:
        return "acoustic_ambient"

    return "other"


# -----------------------
# Feature-based user
# -----------------------

@dataclass
class FeatureUserConfig:
    seed: int = 42
    weight_strength: float = 1.0
    genre_strength: float = 1.0
    global_bias: float = -0.2
    noise: float = 0.0
    top_k_preferred_genres: int = 8

    # Shock settings
    shock_mode: str = "none"  # "none" | "hard" | "gradual"
    shock_at: int = 1000
    shock_window: int = 200  # only for gradual


class FeatureUserSimulator:
    """
    Like ~ Bernoulli(sigmoid(w·z(x) + b_genre + b0)), with optional shock.

    hard shock:
      at t>=shock_at: w := -w_pre, genre_bias := -genre_bias_pre (complete inversion)

    gradual shock:
      over [shock_at, shock_at+shock_window], interpolate from pre to post.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        genre_col: str = "track_genre",
        config: Optional[FeatureUserConfig] = None,
    ) -> None:
        self.config = config or FeatureUserConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.feature_cols = list(feature_cols)
        self.genre_col = genre_col

        # time
        self._turn = 0

        # scaling (only for simulator)
        X = df[self.feature_cols].astype(float).to_numpy()
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        # PRE preferences
        w = self.rng.normal(loc=0.0, scale=1.0, size=len(self.feature_cols))
        w = w / (np.linalg.norm(w) + 1e-12)
        self.w_pre = w * float(self.config.weight_strength)

        # genre bias PRE (optional)
        self.genre_bias_pre: Dict[str, float] = {}
        if self.genre_col in df.columns:
            counts = df[self.genre_col].astype(str).value_counts()
            top_genres = list(counts.index[: max(1, int(self.config.top_k_preferred_genres))])
            for g in counts.index:
                if g in top_genres:
                    self.genre_bias_pre[str(g)] = float(self.rng.normal(loc=+0.8, scale=0.3))
                else:
                    self.genre_bias_pre[str(g)] = float(self.rng.normal(loc=-0.3, scale=0.2))
            for g in list(self.genre_bias_pre.keys()):
                self.genre_bias_pre[g] *= float(self.config.genre_strength)

        self.b0_pre = float(self.config.global_bias)

        # POST preferences (full inversion)
        self.w_post = -self.w_pre
        self.genre_bias_post = {g: -v for g, v in self.genre_bias_pre.items()}
        self.b0_post = self.b0_pre

    def set_turn(self, t: int) -> None:
        self._turn = int(t)

    def _mix(self, pre: float, post: float, s: float) -> float:
        return (1.0 - s) * pre + s * post

    def _shock_strength(self) -> float:
        if self.config.shock_mode == "none":
            return 0.0
        if self.config.shock_mode == "hard":
            return 1.0 if self._turn >= int(self.config.shock_at) else 0.0
        # gradual
        t0 = int(self.config.shock_at)
        W = max(1, int(self.config.shock_window))
        if self._turn < t0:
            return 0.0
        if self._turn >= t0 + W:
            return 1.0
        return float(self._turn - t0) / float(W)

    def p_like(self, song_row: pd.Series) -> float:
        # shock interpolation
        s = self._shock_strength()

        x = song_row[self.feature_cols].astype(float).to_numpy()
        xz = (x - self.mean_) / self.std_

        w = (1.0 - s) * self.w_pre + s * self.w_post
        b0 = self._mix(self.b0_pre, self.b0_post, s)

        lin = float(np.dot(w, xz) + b0)

        if self.genre_col in song_row.index:
            g = str(song_row[self.genre_col])
            gb_pre = float(self.genre_bias_pre.get(g, 0.0))
            gb_post = float(self.genre_bias_post.get(g, 0.0))
            lin += self._mix(gb_pre, gb_post, s)

        if self.config.noise > 0:
            lin += float(self.rng.normal(loc=0.0, scale=self.config.noise))

        return float(sigmoid(lin))

    def rate(self, song_row: pd.Series) -> Tuple[int, float]:
        p = self.p_like(song_row)
        y = int(self.rng.random() < p)
        return y, p


# -----------------------
# Macro-genre probabilistic user
# -----------------------

@dataclass
class MacroGenreProbUserConfig:
    seed: int = 42
    noise: float = 0.0

    # Preference structure
    min_pref: int = 2
    max_pref: int = 5
    p_high: float = 0.85   # P(like | preferred macro)
    p_low: float = 0.15    # P(like | non-preferred macro)

    # Shock
    shock_mode: str = "none"  # "none" | "hard" | "gradual"
    shock_at: int = 1000
    shock_window: int = 200   # only for gradual
    shock_type: str = "invert"  # "invert" | "reroll"


class MacroGenreProbUserSimulator:
    """
    Picks a set of preferred macro-genres.
    Likes are probabilistic:
      if macro in preferred: like ~ Bernoulli(p_high)
      else:                 like ~ Bernoulli(p_low)

    Shock:
      - invert: swap preferred/non-preferred (complete inversion)
      - reroll: draw a new preferred set (new tastes)
    Gradual shock: interpolate p_high/p_low from pre to post over a window.
    """

    def __init__(self, all_genres: Optional[List[str]] = None, config: Optional[MacroGenreProbUserConfig] = None):
        self.config = config or MacroGenreProbUserConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._turn = 0

        # available macros
        macro_set: Set[str] = set()
        if all_genres:
            for g in all_genres:
                macro_set.add(group_genre_to_macro(g))
        if not macro_set:
            macro_set = {
                "metal","classical_like","rock","electronic_dance","hiphop_rnb_soul","jazz_blues",
                "pop_mood","country_folk","latin_brazil","reggae_dancehall","world_regional",
                "east_asia_pop","kids_comedy","soundtrack_show","disco","acoustic_ambient","unknown","other",
            }
        self.macros = sorted(macro_set)

        self.preferred_pre = self._sample_pref_set()
        if self.config.shock_type == "reroll":
            self.preferred_post = self._sample_pref_set()
        else:
            # invert: preferred becomes complement
            self.preferred_post = set(self.macros) - set(self.preferred_pre)

    def _sample_pref_set(self) -> Set[str]:
        k = int(self.rng.integers(int(self.config.min_pref), int(self.config.max_pref) + 1))
        k = max(1, min(k, len(self.macros)))
        return set(self.rng.choice(self.macros, size=k, replace=False).tolist())

    def set_turn(self, t: int) -> None:
        self._turn = int(t)

    def _shock_strength(self) -> float:
        if self.config.shock_mode == "none":
            return 0.0
        if self.config.shock_mode == "hard":
            return 1.0 if self._turn >= int(self.config.shock_at) else 0.0
        t0 = int(self.config.shock_at)
        W = max(1, int(self.config.shock_window))
        if self._turn < t0:
            return 0.0
        if self._turn >= t0 + W:
            return 1.0
        return float(self._turn - t0) / float(W)

    def p_like(self, row: pd.Series) -> float:
        raw = row.get("track_genre", "")
        macro = group_genre_to_macro(raw)
        s = self._shock_strength()

        # preferences shift
        # pre: (preferred_pre, p_high/p_low)
        # post: (preferred_post, p_high/p_low) but swapped when inverted tastes
        if self.config.shock_type == "invert":
            # post probabilities swapped
            p_high_post, p_low_post = self.config.p_low, self.config.p_high
        else:
            p_high_post, p_low_post = self.config.p_high, self.config.p_low

        p_high = (1.0 - s) * self.config.p_high + s * p_high_post
        p_low  = (1.0 - s) * self.config.p_low  + s * p_low_post

        pref = self.preferred_pre if s < 0.5 else self.preferred_post  # set jumps mid-way (ok for our benchmark)

        p = p_high if macro in pref else p_low

        # optional additive noise on probability (small)
        if self.config.noise > 0:
            p = float(np.clip(p + self.rng.normal(0.0, self.config.noise), 0.0, 1.0))
        return float(p)

    def rate(self, row: pd.Series) -> Tuple[int, float]:
        p = self.p_like(row)
        y = int(self.rng.random() < p)
        return y, p




# -----------------------
# Realistic hybrid user
# -----------------------

@dataclass
class RealisticUserConfig:
    """
    A more realistic simulator that mixes:
    - Stable macro-genre preferences (high/low like probabilities by macro)
    - Numeric feature preferences (linear in standardized features)
    - Artist affinity (favorite artists get a boost)
    - Popularity effect (soft threshold)
    - Session moods (context switches every session_len turns)
    - Novelty aversion (too far from recent likes is penalized, with occasional exploration)
    - Fatigue (as time goes on, likes become slightly harder)

    Returns:
      reward ~ Bernoulli(p_true), where p_true is the underlying probability of like.
    """
    seed: int = 42

    # Macro preferences
    n_preferred_macros: int = 3
    macro_p_high: float = 0.75
    macro_p_low: float = 0.15
    macro_logit_weight: float = 1.5  # how strongly macro preference shapes the logit

    # Feature preferences
    weight_strength: float = 1.0  # norm of base feature weight vector

    # Artist affinity
    n_favorite_artists: int = 30
    artist_logit_boost: float = 0.6

    # Popularity (0-100)
    pop_mid: float = 50.0
    pop_slope: float = 0.06  # logistic slope in popularity space
    pop_logit_weight: float = 0.8

    # Sessions / context
    session_len: int = 80  # turns per mood
    mood_logit_scale: float = 0.8  # strength of mood-specific adjustments

    # Novelty / exploration
    novelty_weight: float = 0.8
    exploration_prob: float = 0.08  # chance to ignore novelty penalty

    # Fatigue
    fatigue_per_1000: float = 0.35  # subtract from logit per 1000 turns

    # Observation noise (label flips)
    noise: float = 0.0


class RealisticUserSimulator:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], config: RealisticUserConfig,
                 genre_col: str = "track_genre", artist_col: str = "artists", popularity_col: str = "popularity"):
        self.df = df
        self.feature_cols = list(feature_cols)
        self.genre_col = genre_col
        self.artist_col = artist_col
        self.popularity_col = popularity_col
        self.config = config
        self.rng = np.random.default_rng(int(config.seed))

        # Standardize numeric features for stability
        X = self.df[self.feature_cols].astype(float).to_numpy()
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        # Base feature preference vector (normalized)
        w = self.rng.normal(0.0, 1.0, size=len(self.feature_cols))
        w = w / (np.linalg.norm(w) + 1e-12)
        self.w_base = w * float(self.config.weight_strength)

        # Pick preferred macros (stable taste)
        all_genres = self.df.get(self.genre_col, pd.Series([], dtype=str)).astype(str).unique().tolist()
        all_macros = sorted({group_genre_to_macro(g) for g in all_genres})
        if len(all_macros) == 0:
            all_macros = ["unknown"]
        k = max(1, min(int(self.config.n_preferred_macros), len(all_macros)))
        self.preferred_macros = set(self.rng.choice(all_macros, size=k, replace=False).tolist())

        # Favorite artists (affinity)
        self.favorite_artists = set()
        if self.artist_col in self.df.columns:
            counts = self.df[self.artist_col].astype(str).value_counts()
            top = list(counts.index[: max(1, int(self.config.n_favorite_artists))])
            # Sample favorites from the frequent ones (more realistic than uniform random)
            k_a = min(len(top), int(self.config.n_favorite_artists))
            self.favorite_artists = set(self.rng.choice(top, size=k_a, replace=False).tolist())

        # Mood states: each mood is a small perturbation of weights
        # We keep it simple and plausible: workout -> likes energy/tempo, chill -> acousticness, focus -> instrumentalness
        self.moods = ["workout", "chill", "focus", "party"]
        self.mood_idx = 0
        self.turn = 0

        # Running "taste centroid" from liked songs (for novelty penalty)
        self.like_centroid = np.zeros(len(self.feature_cols), dtype=float)
        self.n_likes = 0

        # Precompute indices for common features (if present)
        self.idx = {c: i for i, c in enumerate(self.feature_cols)}

    def set_turn(self, t: int) -> None:
        self.turn = int(t)
        if self.config.session_len > 0:
            self.mood_idx = (self.turn // int(self.config.session_len)) % len(self.moods)

    def _z(self, row: pd.Series) -> np.ndarray:
        x = row[self.feature_cols].astype(float).to_numpy()
        return (x - self.mean_) / self.std_

    def _mood_weights(self) -> np.ndarray:
        w = self.w_base.copy()
        mood = self.moods[self.mood_idx]

        def bump(name: str, amount: float) -> None:
            if name in self.idx:
                w[self.idx[name]] += amount

        scale = float(self.config.mood_logit_scale)
        if mood == "workout":
            bump("energy", 0.9 * scale)
            bump("tempo", 0.6 * scale)
            bump("danceability", 0.4 * scale)
        elif mood == "chill":
            bump("acousticness", 0.8 * scale)
            bump("instrumentalness", 0.4 * scale)
            bump("energy", -0.5 * scale)
        elif mood == "focus":
            bump("instrumentalness", 0.9 * scale)
            bump("speechiness", -0.4 * scale)
            bump("liveness", -0.3 * scale)
        elif mood == "party":
            bump("danceability", 0.9 * scale)
            bump("valence", 0.6 * scale)
            bump("energy", 0.5 * scale)

        return w

    def _macro_logit(self, row: pd.Series) -> float:
        g = _norm_genre(row.get(self.genre_col, ""))
        macro = group_genre_to_macro(g)
        if macro in self.preferred_macros:
            # map p_high to logit and center around 0
            return float(self.config.macro_logit_weight) * math.log(self.config.macro_p_high / (1.0 - self.config.macro_p_high))
        else:
            return float(self.config.macro_logit_weight) * math.log(self.config.macro_p_low / (1.0 - self.config.macro_p_low))

    def _artist_logit(self, row: pd.Series) -> float:
        if self.artist_col not in row:
            return 0.0
        artist = str(row.get(self.artist_col, ""))
        return float(self.config.artist_logit_boost) if artist in self.favorite_artists else 0.0

    def _popularity_logit(self, row: pd.Series) -> float:
        if self.popularity_col not in row:
            return 0.0
        try:
            pop = float(row.get(self.popularity_col, 0.0))
        except Exception:
            pop = 0.0
        # logistic around pop_mid
        s = float(self.config.pop_slope)
        mid = float(self.config.pop_mid)
        val = 1.0 / (1.0 + math.exp(-s * (pop - mid)))
        # center roughly at 0 by subtracting 0.5
        return float(self.config.pop_logit_weight) * (val - 0.5) * 4.0  # scale to ~[-2,2]

    def _novelty_logit(self, z: np.ndarray) -> float:
        # penalize distance from liked centroid (novelty aversion)
        if self.n_likes <= 5:
            return 0.0
        if self.config.exploration_prob > 0 and self.rng.random() < float(self.config.exploration_prob):
            return 0.0
        centroid = self.like_centroid / max(1, self.n_likes)
        dist = float(np.linalg.norm(z - centroid))
        return -float(self.config.novelty_weight) * dist

    def _fatigue_logit(self) -> float:
        return -float(self.config.fatigue_per_1000) * (float(self.turn) / 1000.0)

    def rate(self, row: pd.Series) -> Tuple[int, float]:
        z = self._z(row)
        w = self._mood_weights()

        logit = 0.0
        logit += float(np.dot(w, z))
        logit += self._macro_logit(row)
        logit += self._artist_logit(row)
        logit += self._popularity_logit(row)
        logit += self._novelty_logit(z)
        logit += self._fatigue_logit()

        # optional gaussian noise on logit (more realistic than label flip)
        # we keep noise as label-flip for compatibility with the rest of your sims
        p_true = float(sigmoid(logit))
        reward = int(self.rng.random() < p_true)

        # optional label flip
        if self.config.noise > 0 and self.rng.random() < float(self.config.noise):
            reward = 1 - reward
            p_true = 1.0 - p_true

        # update centroid if liked (latent preference drift of "what I keep liking")
        if reward == 1:
            self.like_centroid += z
            self.n_likes += 1

        return reward, float(p_true)

# -----------------------
# Random user
# -----------------------

@dataclass
class RandomUserConfig:
    seed: int = 42
    p: float = 0.5


class RandomUserSimulator:
    def __init__(self, config: Optional[RandomUserConfig] = None):
        self.config = config or RandomUserConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._turn = 0

    def set_turn(self, t: int) -> None:
        self._turn = int(t)

    def rate(self, row: pd.Series) -> Tuple[int, float]:
        p = float(self.config.p)
        y = int(self.rng.random() < p)
        return y, p
