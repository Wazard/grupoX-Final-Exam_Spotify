
"""
Simulatore utente "coerente" basato su preferenze latenti:
- pesi su feature numeriche (w)
- bias per track_genre (senza cluster, senza lingua)
Decisione: like ~ Bernoulli(sigmoid(w·x + b_genre + b0))

Questo serve per testare l'algoritmo senza input manuale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def sigmoid(z: float) -> float:
    # robusto per valori grandi/piccoli
    z = float(z)
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)


@dataclass
class UserSimConfig:
    seed: int = 42
    # quanto forti sono le preferenze su feature numeriche
    weight_strength: float = 1.0
    # quanto contano i generi
    genre_strength: float = 1.0
    # bias globale (sposta la like-rate media)
    global_bias: float = -0.2
    # rumore decisionale: 0 = deterministico (soglia 0.5), >0 = probabilistico
    noise: float = 0.0
    # quanti generi "preferiti" selezionare (in base a frequenza nel dataset)
    top_k_preferred_genres: int = 8


class UserSimulator:
    """
    Simulatore utente coerente.
    - costruisce preferenze una volta sola (seed)
    - rating coerente nel tempo
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

        # Fit scaling sui dati disponibili (solo per simulatore)
        X = df[self.feature_cols].astype(float).to_numpy()
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        # Preferenze su feature numeriche
        w = self.rng.normal(loc=0.0, scale=1.0, size=len(self.feature_cols))
        # Normalizzo e applico strength
        w = w / (np.linalg.norm(w) + 1e-12)
        self.w = w * self.config.weight_strength

        # Bias per genere
        self.genre_bias: Dict[str, float] = {}
        if self.genre_col in df.columns:
            counts = df[self.genre_col].astype(str).value_counts()
            top_genres = list(counts.index[: max(1, self.config.top_k_preferred_genres)])
            # preferiti positivi, non preferiti negativi ma più deboli
            for g in counts.index:
                if g in top_genres:
                    self.genre_bias[str(g)] = float(self.rng.normal(loc=+0.8, scale=0.3))
                else:
                    self.genre_bias[str(g)] = float(self.rng.normal(loc=-0.3, scale=0.2))
        # riscalo strength
        for g in list(self.genre_bias.keys()):
            self.genre_bias[g] *= self.config.genre_strength

        self.b0 = float(self.config.global_bias)

    def p_like(self, song_row: pd.Series) -> float:
        x = song_row[self.feature_cols].astype(float).to_numpy()
        xz = (x - self.mean_) / self.std_
        lin = float(np.dot(self.w, xz) + self.b0)

        if self.genre_col in song_row.index:
            g = str(song_row[self.genre_col])
            lin += float(self.genre_bias.get(g, 0.0))

        # noise: aggiungo rumore gaussiano sul logit
        if self.config.noise > 0:
            lin += float(self.rng.normal(loc=0.0, scale=self.config.noise))

        return float(sigmoid(lin))

    def rate(self, song_row: pd.Series) -> Tuple[int, float]:
        """
        Ritorna (label, p_like_true).
        label è campionato da Bernoulli(p_like_true).
        """
        p = self.p_like(song_row)
        y = int(self.rng.random() < p)
        return y, p
