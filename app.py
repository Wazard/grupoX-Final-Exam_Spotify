from recommender.fallback_optimized import generate_fallback_songs # optimized takes roughly 20 times LESS than fallback
from recommender.cold_start import generate_cold_start_songs
from recommender.ranking import generate_ranking
import pandas as pd
import numpy as np
import enum
import time

from features.song_representation import vectorize_song
from user.profile import UserProfile

DATA_PATH = "data/processed/tracks_processed_normalized.csv"
BATCH_SIZE = 20

class App:
    def __init__(self):
        # --- Load data and profile ---
        self.df = pd.read_csv(DATA_PATH)
        self.user_profile = UserProfile.load()
    
    class recommender(enum.Enum):
        COLD_START = 0
        FALLBACK = 1
        RANKING = 2


    # --- Feedback collection ---
    def collect_user_feedback(self, recommendations: pd.DataFrame):
        for _, song in recommendations.iterrows():
            answer = input(
                f"Do you like '{song['track_name']}' by {song['artists']}? (Y/N): "
            ).strip().upper()

            vector, track_id = vectorize_song(song, include_id=True)

            if answer == "Y":
                self.user_profile.like(vector.tolist(), track_id)
            else:
                self.user_profile.dislike(vector.tolist(), track_id)

        self.user_profile.save()


    # --- App decision logic ---
    def choose_recommender(self):
        liked = len(self.user_profile.liked_song_vectors)
        disliked = len(self.user_profile.disliked_song_vectors)

        if liked == 0:
            return self.recommender.COLD_START

        if (liked < 10 and disliked > liked * 1.5) or liked < 10:
            return self.recommender.FALLBACK

        if liked + disliked < 200:
            return self.recommender.RANKING

    # --- Main app loop ---
    def run(self):
        print("\nMusic Recommendation App Started\n")

        while True:
            mode = self.choose_recommender()

            seen_track_ids = set(self.user_profile.liked_song_ids + self.user_profile.disliked_song_ids)
            
            if self.user_profile.has_profile():
                user_vector = np.array(self.user_profile.get_profile_vector())

            start_time = time.perf_counter()

            if mode == self.recommender.COLD_START:
                print("\n--- Cold Start Recommendations ---")
                recommendations = generate_cold_start_songs(self.df, BATCH_SIZE)

            elif mode == self.recommender.FALLBACK:
                print("\n--- Exploration Fallback Recommendations ---")
                recommendations = generate_fallback_songs(
                    df = self.df,
                    liked_vectors = self.user_profile.liked_song_vectors,
                    disliked_vectors = self.user_profile.disliked_song_vectors,
                    seen_track_ids = seen_track_ids,
                    n_songs = BATCH_SIZE
                )
            
            elif mode == self.recommender.RANKING:
                print("\n--- Ranking Recommendations ---")
                recommendations = generate_ranking(
                    df = self.df,
                    user_vector = user_vector,
                    seen_track_ids = seen_track_ids,
                    n_songs = BATCH_SIZE
                )

            elapsed = time.perf_counter() - start_time

            print(f"\n[INFO] Recommendation generated in {elapsed:.3f} seconds")

            self.collect_user_feedback(recommendations)

            cont = input("\nContinue recommending? (Y/N): ").strip().upper()
            if cont != "Y":
                print("\nGoodbye!")
                break
