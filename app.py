from recommender.cold_start import generate_cold_start_songs
from features.song_representation import vectorize_song
from evaluation.metrics import get_similarity
from user.profile import UserProfile
import pandas as pd

FILE_PATH = "data/processed/tracks_processed_normalized.csv"

user_profile = UserProfile.load()
df = pd.read_csv(FILE_PATH, encoding='utf-8')

if not user_profile.has_profile():
    # --- Cold start ---
    cold_start_songs = generate_cold_start_songs(df, 20)

    print("\nCold start songs:")
    print(cold_start_songs[[
        "track_name",
        "artists",
        "track_genre",
        "popularity"
    ]])

    # --- Collect user feedback ---
    for _, song in cold_start_songs.iterrows():
        likeability = input(
            f"Do you like '{song['track_name']}' by {song['artists']}? (Y/N): "
        ).strip().upper()

        if likeability == 'Y':
            user_profile.like(vectorize_song(song))
        else:
            user_profile.dislike(vectorize_song(song))

    user_profile.save()

# --- Similarity sanity check ---
song_a, _ = df.sample(n=2).iloc

vector_song_a = vectorize_song(song_a)
similarity = get_similarity(user_profile.get_profile_vector(), vector_song_a)

print(f"{song_a['track_name']}, by {song_a['artists']} is {similarity} similar to user preferences")