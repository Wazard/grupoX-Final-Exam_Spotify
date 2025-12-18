import pandas as pd
import numpy as np

N_SONGS = 10
POPULARITY_THRESHOLD = 50 
MAX_ARTIST_REPEAT = 2


def generate_cold_start_songs(
    df: pd.DataFrame,
    n_songs: int = N_SONGS,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Cold start recommender.

    Strategy:
    1. Restrict to popular songs (quality)
    2. Identify dominant genres
    3. Pick ONE genre cluster
    4. Sample coherent songs inside that cluster
    """

    rng = np.random.default_rng(random_state)

    # --- Phase 1: quality gate ---
    candidates = df.dropna()
    candidates = candidates[candidates["popularity"] >= POPULARITY_THRESHOLD]

    # --- Phase 2: dominant genres ---
    genre_counts = candidates["track_genre"].value_counts()

    # Keeping top genres only
    top_genres = genre_counts.head(8).index.tolist()

    # --- Phase 3: pick ONE random genre ---
    chosen_genre = rng.choice(top_genres)

    genre_pool = candidates[candidates["track_genre"] == chosen_genre].copy()

    if genre_pool.empty:
        return pd.DataFrame()

    # --- Phase 4: pick a reference song ---
    # Use a strong representative (popular + central)
    ref_song = genre_pool.sample(n=1, random_state=random_state).iloc[0]

    # Use tempo & energy to keep coherence
    tempo_center = ref_song["tempo"]
    energy_center = ref_song["energy"]

    # --- Phase 5: coherence filter ---
    genre_pool["tempo_dist"] = abs(genre_pool["tempo"] - tempo_center)
    genre_pool["energy_dist"] = abs(genre_pool["energy"] - energy_center)

    # Combine distances (simple & interpretable)
    genre_pool["coherence_score"] = (
        genre_pool["tempo_dist"] * 0.5 +
        genre_pool["energy_dist"] * 0.5
    )

    genre_pool = genre_pool.sort_values("coherence_score")

    # --- Phase 6: artist diversity ---
    selected = []
    used_artists = set()

    for _, row in genre_pool.iterrows():
        artist = row["artists"]

        if artist in used_artists:
            continue

        selected.append(row)
        used_artists.add(artist)

        if len(selected) == n_songs:
            break

    return pd.DataFrame(selected).reset_index(drop=True)
