import pandas as pd

N_SONGS = 10
POPULARITY_THRESHOLD = 25


def generate_cold_start_songs(df: pd.DataFrame, n_songs: int = N_SONGS) -> pd.DataFrame:
    """
    Generate a cold-start recommendation list.

    Strategy:
    1. Filter by popularity (quality gate)
    2. Enforce artist uniqueness
    3. Prefer genre diversity
    4. Select exactly n_songs
    """

    # --- Phase 1: candidate pool (quality gate) ---
    candidates = df[df["popularity"] > POPULARITY_THRESHOLD].copy()
    candidates = candidates.dropna()

    # Sort by popularity (descending)
    candidates = candidates.sort_values("popularity", ascending=False)

    # --- Phase 2: enforce artist uniqueness ---
    candidates = (
        candidates
        .groupby("artists", as_index=False)
        .first()
    )

    # Shuffle to avoid popularity-only bias
    candidates = candidates.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Phase 3: final selection with genre diversity ---
    selected_rows = []
    used_artists = set()
    used_genres = set()

    for _, row in candidates.iterrows():
        artist = row["artists"]
        genre = row["track_genre"]

        if artist in used_artists:
            continue

        # Soft genre constraint: allow max 1 per genre
        if genre in used_genres and len(used_genres) < n_songs:
            continue

        selected_rows.append(row)
        used_artists.add(artist)
        used_genres.add(genre)

        if len(selected_rows) == n_songs:
            break

    # Fallback: if genre constraint was too strict
    if len(selected_rows) < n_songs:
        for _, row in candidates.iterrows():
            if row["artists"] in used_artists:
                continue
            selected_rows.append(row)
            used_artists.add(row["artists"])
            if len(selected_rows) == n_songs:
                break

    return pd.DataFrame(selected_rows)
