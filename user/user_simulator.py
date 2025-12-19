from features.song_representation import vectorize_song
from evaluation.metrics import GENRE_CLUSTERS
import pandas as pd
import numpy as np


N_TOTAL_TRACKS = 3000

def simulate_user_feedback(
    recommendations: pd.DataFrame,
    user_profile,
    seed: int = 42,
    like_prob_strong: float = 0.75,
    like_prob_weak: float = 0.25,
    noise_prob: float = 0.05,
):
    """
    Simulate a human user with coherent musical tastes.

    - Strongly likes a few clusters
    - Mostly dislikes others
    - Occasionally behaves inconsistently (noise)

    Parameters
    ----------
    seed : int
        Controls the user's personality
    like_prob_strong : float
        P(like | preferred cluster)
    like_prob_weak : float
        P(like | non-preferred cluster)
    noise_prob : float
        Random behavior chance
    """

    rng = np.random.default_rng(seed)

    # ----------------------------
    # Step 1: choose taste clusters
    # ----------------------------
    cluster_names = list(GENRE_CLUSTERS.keys())

    n_strong = rng.integers(2, 4)  # 2–3 dominant tastes
    strong_clusters = set(rng.choice(cluster_names, size=n_strong, replace=False))

    # Precompute genre → cluster lookup
    genre_to_cluster = {
        genre: cluster
        for cluster, genres in GENRE_CLUSTERS.items()
        for genre in genres
    }

    # ----------------------------
    # Step 2: simulate feedback
    # ----------------------------
    for _, song in recommendations.iterrows():
        
        genre = song["track_genre"]

        print(f"Song: {song['track_name']}, Genre: {genre}")
        
        cluster = genre_to_cluster.get(genre, None)

        vector, track_id = vectorize_song(song, include_id=True)

        # --- Noise (user being random) ---
        if rng.random() < noise_prob:
            like = rng.random() < 0.5

        # --- Strong taste ---
        elif cluster in strong_clusters:
            like = rng.random() < like_prob_strong

        # --- Weak / foreign taste ---
        else:
            like = rng.random() < like_prob_weak

        # ----------------------------
        # Apply feedback
        # ----------------------------
        if like:
            user_profile.like(
                song_vector=vector.tolist(),
                song_id=track_id,
                genre=genre
            )
        else:
            user_profile.dislike(
                song_vector=vector.tolist(),
                song_id=track_id,
                genre=genre
            )

    user_profile.save()
