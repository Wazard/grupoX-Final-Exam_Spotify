import enum
import time
import pandas as pd
from recommender.cold_start import generate_cold_start_songs
from recommender.fallback_optimized import generate_fallback_songs # optimized takes roughly 10 times LESS than fallback

from features.song_representation import vectorize_song
from user.profile import UserProfile


DATA_PATH = "data/processed/tracks_processed_normalized.csv"
BATCH_SIZE = 20

class recommender(enum.Enum):
    COLD_START = "cold_start"
    FALLBACK = "fallback"


# ---------- Load data and profile ----------
df = pd.read_csv(DATA_PATH)
user_profile = UserProfile.load()


# ---------- Feedback collection ----------
def collect_user_feedback(recommendations: pd.DataFrame):
    for _, song in recommendations.iterrows():
        answer = input(
            f"Do you like '{song['track_name']}' by {song['artists']}? (Y/N): "
        ).strip().upper()

        vector, track_id = vectorize_song(song, include_id=True)

        if answer == "Y":
            user_profile.like(vector, track_id)
        else:
            user_profile.dislike(vector, track_id)

    user_profile.save()


# ---------- App decision logic ----------
def choose_recommender():
    liked = len(user_profile.liked_song_vectors)
    disliked = len(user_profile.disliked_song_vectors)

    if liked == 0:
        return recommender.COLD_START

    if liked < 7 and disliked > liked * 1.5:
        return recommender.FALLBACK

    return recommender.FALLBACK

# ---------- Main app loop ----------
def run_app():
    print("\nMusic Recommendation App Started\n")

    while True:
        mode = choose_recommender()

        seen_track_ids = set(
            user_profile.liked_song_ids + user_profile.disliked_song_ids
        )

        start_time = time.perf_counter()

        if mode == recommender.COLD_START:
            print("\n--- Cold Start Recommendations ---")
            recommendations = generate_cold_start_songs(df, BATCH_SIZE)

        elif mode == recommender.FALLBACK:
            print("\n--- Exploration Fallback Recommendations ---")
            recommendations = generate_fallback_songs(
                df=df,
                liked_vectors=user_profile.liked_song_vectors,
                disliked_vectors=user_profile.disliked_song_vectors,
                seen_track_ids=seen_track_ids,
                n_songs=BATCH_SIZE
            )

        elapsed = time.perf_counter() - start_time

        print(f"\n[INFO] Recommendation generated in {elapsed:.3f} seconds")

        collect_user_feedback(recommendations)

        cont = input("\nContinue recommending? (Y/N): ").strip().upper()
        if cont != "Y":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_app()
