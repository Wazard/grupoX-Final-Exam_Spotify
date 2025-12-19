from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class GenreOnlyUserConfig:
    seed: int = 42
    noise: float = 0.0  # 0.0 = deterministico; 0.05 = 5% flip

    # ---- NEW: shock support ----
    enable_shock: bool = False
    shock_at: int = 1000  # "a metà" nel benchmark da 2000 -> 1000 (puoi cambiarlo)

    # Preferenze "prima" dello shock
    like_metal_pre: bool = True
    classical_like_pre: set[str] = field(
        default_factory=lambda: {"classical", "opera", "piano", "new-age"}
    )

    # Preferenze "dopo" lo shock (default: cambia gusto -> NON gli piace più metal/classical_like)
    like_metal_post: bool = False
    classical_like_post: set[str] = field(default_factory=set)


class GenreOnlyUserSimulator:
    """
    Utente deterministico (o quasi) basato SOLO sul track_genre.

    Modalità base (enable_shock=False):
      Like se:
        - 'metal' è substring del genere
        - oppure genere in un set "classico/simile"

    Modalità shock (enable_shock=True):
      - prima di shock_at usa preferenze *_pre
      - da shock_at in poi usa preferenze *_post
      - serve chiamare set_turn(t) dal loop/benchmark per farlo funzionare
    """

    def __init__(self, config: GenreOnlyUserConfig | None = None):
        self.config = config or GenreOnlyUserConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._turn: int = 0  # turn corrente (gestito dal benchmark)

    # ---- NEW: turn hook (chiamalo nel benchmark prima di rate) ----
    def set_turn(self, t: int) -> None:
        self._turn = int(t)

    def _current_prefs(self) -> tuple[bool, set[str]]:
        if self.config.enable_shock and self._turn >= self.config.shock_at:
            return self.config.like_metal_post, self.config.classical_like_post
        return self.config.like_metal_pre, self.config.classical_like_pre

    def rate(self, row: pd.Series) -> tuple[int, float]:
        genre = str(row.get("track_genre", "")).strip().lower()

        like_metal, classical_like = self._current_prefs()

        like = 0
        if like_metal and ("metal" in genre):
            like = 1
        elif genre in classical_like:
            like = 1

        p_true = 1.0 if like == 1 else 0.0

        # Rumore opzionale: flip con prob noise
        if self.config.noise > 0 and self.rng.random() < self.config.noise:
            like = 1 - like
            p_true = 1.0 - p_true

        return int(like), float(p_true)
