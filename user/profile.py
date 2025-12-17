from typing import List
import numpy as np
import json
import os

PROFILE_PATH = "user/user_profile.json"
DISLIKE_WEIGHT = 0.3


class UserProfile:
    def __init__(self):
        self.liked_song_vectors: List[List[float]] = []
        self.disliked_song_vectors: List[List[float]] = []
        self.liked_song_ids: list[str] = []
        self.disliked_song_ids: list[str] = []

    # ---------- feedback ----------
    def like(self, song_vector: List[float], song_id: str = None):
        self.liked_song_vectors.append(song_vector)
        if song_id is not None:
            self.liked_song_ids.append(song_id)

    def dislike(self, song_vector: List[float], song_id: str = None):
        self.disliked_song_vectors.append(song_vector)
        if song_id is not None:
            self.disliked_song_ids.append(song_id)
    

    # ---------- profile computation ----------
    def has_profile(self) -> bool:
        return len(self.liked_song_vectors) > 0

    def get_profile_vector(self) -> List[float]:
        if not self.has_profile():
            raise ValueError("User profile is empty (no liked songs).")
        if self.liked_song_vectors:
            positive = np.mean(np.array(self.liked_song_vectors), axis=0)

        if self.disliked_song_vectors:
            negative = np.mean(np.array(self.disliked_song_vectors), axis=0)
            profile = positive - DISLIKE_WEIGHT * negative
        else:
            profile = positive

        return profile.tolist()

    # ---------- persistence ----------
    def save(self, path: str = PROFILE_PATH):
        data = {
            "liked_vectors": self.liked_song_vectors,
            "disliked_vectors": self.disliked_song_vectors
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str = PROFILE_PATH):
        profile = cls()

        if not os.path.exists(path):
            return profile  # empty profile

        with open(path, "r") as f:
            data = json.load(f)

        profile.liked_song_vectors = data.get("liked_vectors", [])
        profile.disliked_song_vectors = data.get("disliked_vectors", [])

        return profile