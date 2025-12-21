from visualization.visualize_plots import plot_active_profile_vectors_from_json, plot_global_genre_radar_from_json
from evaluation.metrics import SIMILARITY_FEATURES
from dotenv import load_dotenv
from app import App
import time

PLOT_ONLY = False
PROFILE_PATH = "user/user_profile.json"
DELAY = 2.5

if PLOT_ONLY and PROFILE_PATH:
    plot_active_profile_vectors_from_json(
        profile_json_path= PROFILE_PATH,
        feature_names=SIMILARITY_FEATURES
        )
    plot_global_genre_radar_from_json(profile_json_path= PROFILE_PATH)


if __name__ == "__main__" and not PLOT_ONLY:
    load_dotenv()
    app = App()
    app.run()
    while True:
        if not app.is_running:
            time.sleep(DELAY) 
            plot_active_profile_vectors_from_json(
                profile_json_path= PROFILE_PATH,
                feature_names=SIMILARITY_FEATURES
                )
            plot_global_genre_radar_from_json(profile_json_path= PROFILE_PATH)
            break;