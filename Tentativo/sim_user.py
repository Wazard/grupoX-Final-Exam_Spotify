"""
Simulatore utente "coerente" basato su preferenze latenti:
- pesi su feature numeriche (w)
- bias per track_genre (senza cluster, senza lingua)
Decisione: like ~ Bernoulli(sigmoid(w·x + b_genre + b0))

Questo serve per testare l'algoritmo senza input manuale.

NEW:
- flag shock: dopo shock_at iterazioni, cambia COMPLETAMENTE gusti (inversione)
- per farlo funzionare devi chiamare user.set_turn(t) nel loop prima di rate()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)


@dataclass
class UserSimConfig:
    seed: int = 42
    weight_strength: float = 1.0
    genre_strength: float = 1.0
    global_bias: float = -0.2
    noise: float = 0.0
    top_k_preferred_genres: int = 8

    # -------------------
    # SHOCK SETTINGS (NEW)
    # -------------------
    enable_shock: bool = False
    shock_at: int = 1000  # iterazione (turno) in cui scatta lo shock


class UserSimulator:
    """
    Simulatore utente coerente.
    - costruisce preferenze una volta sola (seed)
    - rating coerente nel tempo
    - opzionale shock completo (inversione gusti) dopo shock_at
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        genre_col: str = "track_genre",
        config: Optional[UserSimConfig] = None,
    ) -> None:
        self.config = config or UserSimConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.feature_cols = list(feature_cols)
        self.genre_col = genre_col

        # stato shock
        self._turn: int = 0
        self._shock_applied: bool = False

        # Fit scaling sui dati disponibili (solo per simulatore)
        X = df[self.feature_cols].astype(float).to_numpy()
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        # Preferenze PRE (prima dello shock)
        w = self.rng.normal(loc=0.0, scale=1.0, size=len(self.feature_cols))
        w = w / (np.linalg.norm(w) + 1e-12)
        self.w_pre = w * float(self.config.weight_strength)

        # Bias per genere PRE
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

        # Preferenze POST (shock completo = inversione)
        self.w_post = -self.w_pre
        self.genre_bias_post = {g: -v for g, v in self.genre_bias_pre.items()}
        self.b0_post = self.b0_pre  # puoi anche cambiarlo se vuoi spostare la like-rate media

        # default: PRE
        self.w = self.w_pre
        self.genre_bias = self.genre_bias_pre
        self.b0 = self.b0_pre

    # ----------------
    # SHOCK HOOK (NEW)
    # ----------------
    def set_turn(self, t: int) -> None:
        """Chiamalo nel loop (benchmark/main) prima di rate()."""
        self._turn = int(t)
        self._maybe_apply_shock()

    def _maybe_apply_shock(self) -> None:
        if not self.config.enable_shock:
            return
        if (not self._shock_applied) and (self._turn >= int(self.config.shock_at)):
            self.w = self.w_post
            self.genre_bias = self.genre_bias_post
            self.b0 = self.b0_post
            self._shock_applied = True

    def p_like(self, song_row: pd.Series) -> float:
        x = song_row[self.feature_cols].astype(float).to_numpy()
        xz = (x - self.mean_) / self.std_
        lin = float(np.dot(self.w, xz) + self.b0)

        if self.genre_col in song_row.index:
            g = str(song_row[self.genre_col])
            lin += float(self.genre_bias.get(g, 0.0))

        if self.config.noise > 0:
            lin += float(self.rng.normal(loc=0.0, scale=self.config.noise))

        return float(sigmoid(lin))

    def rate(self, song_row: pd.Series) -> Tuple[int, float]:
        p = self.p_like(song_row)
        y = int(self.rng.random() < p)
        return y, p
