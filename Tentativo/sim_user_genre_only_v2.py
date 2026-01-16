from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------
# Genre grouping utilities
# -----------------------

def _norm_genre(g: str) -> str:
    return str(g or "").strip().lower()


def group_genre_to_macro(genre: str) -> str:
    """
    Raggruppa track_genre (granulare) in macro-generi più grandi.
    L'obiettivo è coprire TUTTI i generi del dataset, anche quelli rari,
    usando regole robuste (substring/pattern).
    """
    g = _norm_genre(genre)
    if g == "":
        return "unknown"

    # METAL (tutte le varianti)
    if "metal" in g or g in {"grindcore", "hardcore"}:
        return "metal"

    # CLASSICAL / ORCHESTRAL / NEAR-CLASSICAL
    if g in {"classical", "opera", "piano", "new-age"}:
        return "classical_like"
    if g in {"show-tunes", "disney"}:
        return "soundtrack_show"

    # ROCK family
    if "rock" in g or g in {"grunge", "emo", "punk", "ska", "goth", "guitar"} or "punk" in g:
        return "rock"

    # ELECTRONIC / DANCE
    if any(k in g for k in ["techno", "house", "edm", "dub", "dubstep", "electro", "electronic", "trance",
                            "drum-and-bass", "breakbeat", "garage", "hardstyle", "club", "minimal-techno", "idm",
                            "deep-house", "detroit-techno", "chicago-house", "progressive-house", "trip-hop"]):
        return "electronic_dance"

    # HIP-HOP / R&B / SOUL
    if g in {"hip-hop", "r-n-b", "soul", "funk"}:
        return "hiphop_rnb_soul"

    # JAZZ / BLUES
    if g in {"jazz", "blues"}:
        return "jazz_blues"

    # POP (incl. synth-pop / indie-pop / power-pop)
    if g in {"pop", "synth-pop", "power-pop", "indie-pop", "pop-film", "party", "happy", "sad", "sleep", "study"}:
        return "pop_mood"

    # COUNTRY / FOLK / SINGER-SONGWRITER
    if g in {"country", "honky-tonk", "bluegrass", "folk", "singer-songwriter", "songwriter"}:
        return "country_folk"

    # LATIN / BRAZIL / IBERIA
    if g in {"latin", "latino", "reggaeton", "salsa", "samba", "brazil", "mpb", "forro", "pagode", "sertanejo",
             "spanish", "romance"}:
        return "latin_brazil"

    # REGGAE / DANCEHALL
    if g in {"reggae", "dancehall"}:
        return "reggae_dancehall"

    # WORLD / REGIONAL / LANGUAGE tags
    if g in {"world-music", "turkish", "iranian", "indian", "malay", "german", "french", "swedish", "british",
             "cantopop", "mandopop"}:
        return "world_regional"
    if g in {"j-pop", "j-rock", "j-idol", "j-dance", "k-pop", "anime"}:
        return "east_asia_pop"

    # CHILDREN / KIDS / COMEDY
    if g in {"children", "kids", "comedy"}:
        return "kids_comedy"

    # DISCO / SOFT buckets
    if g in {"disco"}:
        return "disco"

    # ACOUSTIC / AMBIENT / CHILL
    if g in {"acoustic", "ambient", "chill"}:
        return "acoustic_ambient"

    # Default fallback: keep the original tag but marked as other
    return "other"


# -----------------------
# Simulator
# -----------------------

@dataclass
class GenreOnlyUserConfig:
    seed: int = 42
    noise: float = 0.5 # 0.0 = deterministico; 0.05 = 5% flip
    # range of preferred macro-genres
    min_likes: int = 1


class GenreOnlyUserSimulator:
    """
    Utente basato SOLO sul track_genre, ma con:
    - raggruppamento in macro-generi (per coprire tutto il dataset)
    - numero casuale di macro-generi preferiti (bias verso 2..5)
    - scelta casuale dei macro-generi preferiti
    """

    def __init__(self, all_genres: List[str] | None = None, config: GenreOnlyUserConfig | None = None):
        self.config = config or GenreOnlyUserConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Costruisco elenco macro-generi disponibili
        macro_set: Set[str] = set()
        if all_genres:
            for g in all_genres:
                macro_set.add(group_genre_to_macro(g))
        # fallback (se non passo all_genres)
        if not macro_set:
            macro_set = {
                "metal",
                "classical_like",
                "rock",
                "electronic_dance",
                "hiphop_rnb_soul",
                "jazz_blues",
                "pop_mood",
                "country_folk",
                "latin_brazil",
                "reggae_dancehall",
                "world_regional",
                "east_asia_pop",
                "kids_comedy",
                "soundtrack_show",
                "disco",
                "acoustic_ambient",
                "unknown",
                "other",
            }

        self.macro_genres = sorted(macro_set)
        self.n_macro = len(self.macro_genres)

        # Scegli quanti macro-generi preferiti: 1..n_macro con più massa su 2..5
        # Costruisco pesi: 2..5 molto più probabili, poi decresce.
        ks = np.arange(1, self.n_macro + 1)
        weights = np.ones_like(ks, dtype=float)
        for i, k in enumerate(ks):
            if 2 <= k <= 5:
                weights[i] = 6.0
            elif k == 1:
                weights[i] = 2.5
            else:
                weights[i] = 1.0 / (1.0 + 0.15 * (k - 5))  # decay soft
        weights = weights / weights.sum()

        k_pref = int(self.rng.choice(ks, p=weights))
        k_pref = max(int(self.config.min_likes), min(k_pref, self.n_macro))

        self.preferred_macros: Set[str] = set(self.rng.choice(self.macro_genres, size=k_pref, replace=False).tolist())

    def rate(self, row: pd.Series) -> Tuple[int, float]:
        genre = _norm_genre(row.get("track_genre", ""))
        macro = group_genre_to_macro(genre)

        like = 1 if macro in self.preferred_macros else 0
        p_true = 1.0 if like == 1 else 0.0


        # Rumore opzionale: flip con prob noise
        if self.config.noise > 0 and self.rng.random() < self.config.noise:
            like = 1 - like
            p_true = 1.0 - p_true

        return int(like), float(p_true)
