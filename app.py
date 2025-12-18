from recommender.model_data import (
    generate_model_rank,
    build_training_data,
    train_like_model,
)
from features.spotify_api import SpotifyTokenManager, get_spotify_links_and_images
from recommender.fallback_optimized import generate_fallback_songs
from recommender.new_cold_start import generate_cold_start_songs
from recommender.ranking import generate_ranking
from features.song_representation import vectorize_song
from user.profile import UserProfile

import pandas as pd
import numpy as np
import pyperclip
import enum
import time


DATA_PATH = "data/processed/tracks_processed_normalized.csv"
BATCH_SIZE = 10  # must be <= 50 for Spotify batch endpoints

class App:
    class Recommender(enum.Enum):
        COLD_START = 0
        FALLBACK = 1
        RANKING = 2
        MODEL = 3

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.df["track_id"] = self.df["track_id"].astype(str).str.strip()

        self.user_profile = UserProfile.load()
        self.token_manager = SpotifyTokenManager()

        self.model = None
        self.train_data = None
        self.last_model_train_seen = 0

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def get_recommendations_with_urls_img(
        self,
        recommendations: pd.DataFrame,
        spotify_token: str
    ) -> pd.DataFrame:
        """
        Attach Spotify external URLs and album images to recommendations.
        Safe against missing or partial API responses.
        """
        if recommendations.empty:
            return recommendations

        tmp = recommendations.copy()

        track_ids = tmp["track_id"].astype(str).tolist()

        links_images = get_spotify_links_and_images(
            track_ids=track_ids,
            spotify_token=spotify_token
        )

        tmp["track_url"] = tmp["track_id"].map(
            lambda tid: links_images.get(tid, {}).get("spotify_url")
        )

        tmp["album_image"] = tmp["track_id"].map(
            lambda tid: links_images.get(tid, {}).get("image_url")
        )

        return tmp

    # -------------------------------------------------
    # Recommender choice
    # -------------------------------------------------
    def choose_recommender(self):
        c = self.user_profile.confidence

        if c < 2:
            return self.Recommender.COLD_START
        if c < 4:
            return self.Recommender.FALLBACK
        if c < 10:
            return self.Recommender.RANKING
        return self.Recommender.MODEL

    # -------------------------------------------------
    # Feedback loop
    # -------------------------------------------------
    def collect_user_feedback(self, recommendations: pd.DataFrame):
        i = 0
        n = len(recommendations)

        while i < n:
            song = recommendations.iloc[i]
            genre = song.get("track_genre")

            if song.get("track_url"):
                pyperclip.copy(song["track_url"])

            answer = input(
                f"({i+1}/{n}) Do you like '{song['track_name']}' by {song['artists']}? "
                "[Y/N]: "
            ).strip().upper()

            vector, track_id = vectorize_song(song, include_id=True)

            if answer == "Y":
                self.user_profile.like(
                    song_vector=vector.tolist(),
                    song_id=track_id,
                    genre=genre
                )
            else:
                self.user_profile.dislike(
                    song_vector=vector.tolist(),
                    song_id=track_id,
                    genre=genre
                )

            i += 1

        self.user_profile.save()

    # -------------------------------------------------
    # Deduplication
    # -------------------------------------------------
    def _dedup_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate songs inside a single recommendation batch.
        Deduplication is semantic, not by track_id.

        Keeps the first occurrence (already sorted by score upstream).
        """
        if df.empty:
            return df

        dedup_cols = ["track_name", "artists", "duration_s"]

        # Some datasets may lack duration_s – fallback safely
        existing_cols = [c for c in dedup_cols if c in df.columns]

        if not existing_cols:
            return df.drop_duplicates(subset=["track_id"])

        return (
            df
            .drop_duplicates(subset=existing_cols, keep="first")
            .reset_index(drop=True)
        )

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def run(self):
        print("\nMusic Recommendation App Started\n")

        while True:
            mode = self.choose_recommender()

            start = time.perf_counter()

            if mode == self.Recommender.COLD_START:
                print("\n--- Cold Start ---")
                recs = generate_cold_start_songs(self.df, BATCH_SIZE)

            elif mode == self.Recommender.FALLBACK:
                print("\n--- Fallback ---")
                recs = generate_fallback_songs(
                    df=self.df,
                    user_profile=self.user_profile,
                    n_songs=BATCH_SIZE,
                )

            elif mode == self.Recommender.RANKING:
                print("\n--- Ranking ---")
                recs = generate_ranking(
                    df=self.df,
                    user_profile=self.user_profile,
                    n_songs=BATCH_SIZE,
                )

            else:
                print("\n--- Model ---")

                if self.model is None:
                    self.train_data = build_training_data(
                        df=self.df,
                        liked_ids=self.user_profile.liked_song_ids,
                        disliked_ids=self.user_profile.disliked_song_ids,
                    )
                    self.model = train_like_model(*self.train_data)
                    self.last_model_train_seen = len(self.user_profile.seen_song_ids)

                recs = generate_model_rank(
                    df=self.df,
                    model=self.model,
                    seen_track_ids=self.user_profile.seen_song_ids,
                    n_songs=BATCH_SIZE,
                )

            elapsed = time.perf_counter() - start
            print(f"[INFO] Generated in {elapsed:.3f}s")

            recs = self._dedup_batch(recs)
            spotify_token = self.token_manager.get_token()
            recs = self.get_recommendations_with_urls_img(recs, spotify_token)

            self.collect_user_feedback(recs)

            if input("\nContinue? (Y/N): ").upper() != "Y":
                break
