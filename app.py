from features.spotify_api           import get_spotify_links_and_images, SpotifyTokenManager
from recommender.model_data         import generate_linear_model_rank, train_linear_models_if_needed
from recommender.light_gbm_new      import generate_lgbm_rank, train_lgbms_if_needed
from user.user_simulator            import simulate_user_feedback, N_TOTAL_TRACKS
from recommender.fallback_optimized import generate_fallback_songs
from recommender.new_cold_start     import generate_cold_start_songs
from features.song_representation   import vectorize_song
from recommender.ranking            import generate_ranking
from user.profile                   import UserProfile

import pandas       as pd
import pyperclip    as clip

import enum
import time

# simulate user
SIMULATE_USER = True

# Configs
DATA_PATH = "data/processed/tracks_processed_normalized.csv"
BATCH_SIZE = 10             # must be <= 50 for Spotify batch endpoints
COLD_START_BATCH_MUL = 2    # multiplier for cold start batch size, 1 = batch_size, 2 = 2*batch_size
SIMULATED_USER_SEED = 262

# CAPS
COLD_START_BATCH_MUL = min(
    BATCH_SIZE*COLD_START_BATCH_MUL, 50
    )/BATCH_SIZE                            # caps cold start batch size at 50 
BATCH_SIZE = min(BATCH_SIZE, 50)            # caps batch size at 50

class App:
    class Recommender(enum.Enum):
        COLD_START = 0
        FALLBACK = 1
        RANKING = 2
        LINEAR_MODEL = 3
        BOOST_MODEL = 4

    def __init__(self):
        # App variables
        self.df = pd.read_csv(DATA_PATH)
        self.df["track_id"] = self.df["track_id"].astype(str).str.strip()
        self.user_profile = UserProfile.load()

        # App state
        self.token_manager = SpotifyTokenManager()
        self.is_running = False

        # Train model data
        self.train_data = None
        self.linear_model = None
        self.boost_model = None
        self.last_linear_model_train_seen = 0
        self.last_boost_model_train_seen = 0

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
        '''
        Choses recommender based on confidence and seen songs amount
        '''
        c = self.user_profile.confidence
        n = len(self.user_profile.seen_song_ids)
        print(f"total confidence {c}")

        if c < 2:
            return self.Recommender.COLD_START
        if c < 4:
            return self.Recommender.FALLBACK
        if c < 8 or n < 200:
            return self.Recommender.RANKING
        if c < 12 or n < 2000:
            return self.Recommender.LINEAR_MODEL
        
        return self.Recommender.BOOST_MODEL
    # -------------------------------------------------
    # Feedback loop
    # -------------------------------------------------

    def collect_user_feedback(self, recommendations: pd.DataFrame):

        '''
        Asks the user if he likes or not the recommended songs
        
        :param self: Description
        :param recommendations: Description
        :type recommendations: pd.DataFrame
        '''
        i = 0
        n = len(recommendations)

        while i < n:
            song = recommendations.iloc[i]
            genre = song.get("track_genre")

            if song.get("track_url"):
                clip.copy(song["track_url"])

            answer = input(
                f"({i+1}/{n}) Do you like: {song['track_name']}, by {song['artists']}, of genre: {song['track_genre']}? [Y/N]: "
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
    # Main loop
    # -------------------------------------------------

    def run(self):
        '''
        Runs the application
        '''
        self.is_running = True
        print("\nMusic Recommendation App Started\n")

        while True:
            mode = self.choose_recommender()

            start = time.perf_counter()

            if mode == self.Recommender.COLD_START:
                print("\n--- Cold Start ---")
                recommended_tracks = generate_cold_start_songs(self.df, BATCH_SIZE * COLD_START_BATCH_MUL)

            elif mode == self.Recommender.FALLBACK:
                print("\n--- Fallback ---")
                recommended_tracks = generate_fallback_songs(
                    df=self.df,
                    user_profile=self.user_profile,
                    seen_track_ids=self.user_profile.seen_song_ids,
                    n_songs=BATCH_SIZE,
                )

            elif mode == self.Recommender.RANKING:
                print("\n--- Ranking ---")
                recommended_tracks = generate_ranking(
                    df=self.df,
                    user_profile=self.user_profile,
                    n_songs=BATCH_SIZE,
                )

            elif mode == self.Recommender.LINEAR_MODEL:
                print("\n--- Linear Model ---")

                # Train / refresh per-taste models only when needed
                self.last_linear_model_train_seen = train_linear_models_if_needed(
                    df=self.df,
                    user_profile=self.user_profile,
                    last_train_seen=self.last_linear_model_train_seen,
                )

                # Generate recommendations using ALL taste profiles
                recommended_tracks = generate_linear_model_rank(
                    df=self.df,
                    user_profile=self.user_profile,
                    seen_track_ids=self.user_profile.seen_song_ids,
                    n_songs=BATCH_SIZE,
                )
            
            elif mode == self.Recommender.BOOST_MODEL:
                print("\n--- Boost Model ---")
            
                self.last_boost_model_train_seen = train_lgbms_if_needed(
                    df = self.df,
                    user_profile= self.user_profile,
                    last_train_seen=self.last_boost_model_train_seen,
                )

                recommended_tracks = generate_lgbm_rank(
                    df = self.df,
                    user_profile=self.user_profile,
                    seen_track_ids= self.user_profile.seen_song_ids,
                    n_songs=BATCH_SIZE
                )

            elapsed = time.perf_counter() - start
            print(f"[INFO] Generated in {elapsed:.3f}s")

            spotify_token = self.token_manager.get_token()
            recommended_tracks = self.get_recommendations_with_urls_img(recommended_tracks, spotify_token)
            
            simulate_user_feedback(recommended_tracks, self.user_profile, SIMULATED_USER_SEED)

            if not SIMULATE_USER:
                self.collect_user_feedback(recommended_tracks)

                if input("\nContinue? (Y/N): ").upper() != "Y":
                    break
            
            total_seen = len(self.user_profile.seen_song_ids)
            print(f"\ntotal seen tracks: {total_seen}\n")
            if  total_seen>= N_TOTAL_TRACKS:
                self.is_running = False
                break

