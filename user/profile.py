import numpy as np
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional

from evaluation.metrics import GENRE_CLUSTERS

# ============================================================
# Configuration
# ============================================================
PROFILE_PATH = "user/user_profile.json"

SIM_THRESHOLD_ASSIGN = 0.6
TIME_FACTOR = 0.08
DISLIKE_PENALTY = 0.15


# ============================================================
# Taste Profile (one genre cluster)
# ============================================================
@dataclass
class TasteProfile:
    cluster_name: str
    genres: set[str]

    vector: np.ndarray
    confidence: float
    liked_count: int
    genre_counts: Counter

    def normalize(self):
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm


# ============================================================
# User Profile
# ============================================================
class UserProfile:
    def __init__(self):
        self.taste_profiles: List[TasteProfile] = []

        self.liked_song_ids: List[str] = []
        self.disliked_song_ids: List[str] = []

        # Pre-generate empty profiles from genre clusters
        self._init_profiles_from_clusters()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def _init_profiles_from_clusters(self):
        """
        Create one empty taste profile per genre cluster.
        """
        for cluster_name, genres in GENRE_CLUSTERS.items():
            self.taste_profiles.append(
                TasteProfile(
                    cluster_name=cluster_name,
                    genres=set(genres),
                    vector=np.zeros(len(genres), dtype=np.float32),  # placeholder
                    confidence=0.0,
                    liked_count=0,
                    genre_counts=Counter()
                )
            )

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------
    @property
    def seen_song_ids(self):
        return set(self.liked_song_ids + self.disliked_song_ids)

    @property
    def confidence(self) -> float:
        return sum(p.confidence for p in self.taste_profiles)

    def has_profile(self) -> bool:
        return any(p.liked_count > 0 for p in self.taste_profiles)

    # --------------------------------------------------
    # Feedback
    # --------------------------------------------------
    def like(self, song_vector: List[float], song_id: str, genre: str):
        vec = np.array(song_vector, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-8

        self.liked_song_ids.append(song_id)

        profile = self._profile_for_genre(genre)
        if profile is None:
            return

        # First like initializes vector
        if profile.liked_count == 0:
            profile.vector = vec.copy()
        else:
            profile.vector = (
                (1 - TIME_FACTOR) * profile.vector +
                TIME_FACTOR * vec
            )

        profile.normalize()
        profile.liked_count += 1
        profile.genre_counts[genre] += 1
        profile.confidence = np.log1p(profile.liked_count)

    def dislike(self, song_vector: List[float], song_id: str, genre: str):
        vec = np.array(song_vector, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-8

        self.disliked_song_ids.append(song_id)

        profile = self._profile_for_genre(genre)
        if profile is None or profile.liked_count == 0:
            return

        profile.confidence *= (1.0 - DISLIKE_PENALTY)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _profile_for_genre(self, genre: str) -> Optional[TasteProfile]:
        for p in self.taste_profiles:
            if genre in p.genres:
                return p
        return None

    def get_active_profiles(self, min_confidence: float = 1.0) -> List[TasteProfile]:
        """
        Profiles that are actually meaningful.
        """
        return [
            p for p in self.taste_profiles
            if p.confidence >= min_confidence and p.liked_count > 0
        ]

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self, path: str = PROFILE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "liked_song_ids": self.liked_song_ids,
            "disliked_song_ids": self.disliked_song_ids,
            "taste_profiles": [
                {
                    "cluster_name": p.cluster_name,
                    "vector": p.vector.tolist(),
                    "confidence": p.confidence,
                    "liked_count": p.liked_count,
                    "genre_counts": dict(p.genre_counts),
                }
                for p in self.taste_profiles
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str = PROFILE_PATH):
        profile = cls()

        if not os.path.exists(path):
            return profile

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile.liked_song_ids = data.get("liked_song_ids", [])
        profile.disliked_song_ids = data.get("disliked_song_ids", [])

        saved_profiles = {
            p["cluster_name"]: p
            for p in data.get("taste_profiles", [])
        }

        for p in profile.taste_profiles:
            saved = saved_profiles.get(p.cluster_name)
            if not saved:
                continue

            p.vector = np.array(saved["vector"], dtype=np.float32)
            p.confidence = float(saved["confidence"])
            p.liked_count = int(saved["liked_count"])
            p.genre_counts = Counter(saved["genre_counts"])

        return profile
