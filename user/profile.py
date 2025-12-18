import numpy as np
import json
import os

PROFILE_PATH = "user/user_profile.json"
DISLIKE_WEIGHT = 0.01    # How much disliked songs count on user_vector
TIME_FACTOR = 0.04      # how much does newer feedback weight on older songs (.01 means: old_profile=96%, new_song = 4%)


class UserProfile:
    def __init__(self):
        self.liked_song_vectors: list[list[float]] = []
        self.disliked_song_vectors: list[list[float]] = []
        self.liked_song_ids: list[str] = []
        self.disliked_song_ids: list[str] = []

        # Always keep this normalized
        self.profile_vector: list[float] = []

    # --------------------------------------------------
    # Feedback
    # --------------------------------------------------
    def like(self, song_vector: list[float], song_id: str = None):
        self.liked_song_vectors.append(song_vector)
        if song_id is not None:
            self.liked_song_ids.append(song_id)

        self._update_with_vector(np.array(song_vector), positive=True)

    def dislike(self, song_vector: list[float], song_id: str = None):
        self.disliked_song_vectors.append(song_vector)
        if song_id is not None:
            self.disliked_song_ids.append(song_id)

        self._update_with_vector(np.array(song_vector), positive=False)

    # --------------------------------------------------
    # Profile logic (incremental)
    # --------------------------------------------------
    def has_profile(self) -> bool:
        return len(self.profile_vector) > 0

    def _update_with_vector(self, vec: np.ndarray, positive: bool):
        """
        Incrementally update the profile vector.
        """
        if not self.profile_vector:
            # First liked song initializes the profile
            profile = vec.copy()
        else:
            profile = np.array(self.profile_vector)

            if positive:
                profile = (1 - TIME_FACTOR) * profile + TIME_FACTOR * vec
            else:
                profile = profile - TIME_FACTOR * DISLIKE_WEIGHT * vec

        # Normalize (direction matters, magnitude does not)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        self.profile_vector = profile.tolist()

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self, path: str = PROFILE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "liked_vectors": self.liked_song_vectors,
            "disliked_vectors": self.disliked_song_vectors,
            "liked_song_ids": self.liked_song_ids,
            "disliked_song_ids": self.disliked_song_ids,
            "profile_vector": self.profile_vector,
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

        profile.liked_song_vectors = data.get("liked_vectors", [])
        profile.disliked_song_vectors = data.get("disliked_vectors", [])
        profile.liked_song_ids = data.get("liked_song_ids", [])
        profile.disliked_song_ids = data.get("disliked_song_ids", [])
        profile.profile_vector = data.get("profile_vector", [])

        return profile
