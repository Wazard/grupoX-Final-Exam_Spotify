import numpy as np

SIMILARITY_FEATURES = np.array(['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness'])
BANNED_GENRES = {"sleep", "white-noise", "ambient", "lullaby", "nature", "background"}
FEATURE_WEIGHTS = {
    'danceability': 1.0, 'energy': 1.0, 'valence': 0.9, 'tempo': 1.1, 
    'loudness': 1.0, 'acousticness': 0.8, 'instrumentalness': 1.0, 
    'liveness': 0.9, 'speechiness': 1.0
}
WEIGHTS = np.array([FEATURE_WEIGHTS[f] for f in SIMILARITY_FEATURES])
GENRE_CLUSTERS = {
    "mainstream_pop_dance": [
        "pop", "dance", "edm", "club", "disco", "party",
        "synth-pop", "indie-pop", "power-pop", "pop-film", "disney"
    ],

    "electronic_club": [
        "house", "deep-house", "progressive-house", "chicago-house",
        "techno", "minimal-techno", "detroit-techno", "trance",
        "hardstyle", "electro", "electronic", "idm",
        "garage", "breakbeat", "drum-and-bass", "dub", "dubstep"
    ],

    "rock": [
        "rock", "alt-rock", "alternative", "hard-rock",
        "rock-n-roll", "rockabilly", "psych-rock",
        "grunge", "indie", "british"
    ],

    "metal_extreme": [
        "metal", "heavy-metal", "death-metal",
        "black-metal", "metalcore", "grindcore",
        "hardcore", "industrial"
    ],

    "punk_emo_goth": [
        "punk", "punk-rock", "emo", "goth"
    ],

    "hiphop_rnb_reggae": [
        "hip-hop", "r-n-b", "soul", "funk", "groove",
        "reggae", "reggaeton", "dancehall", "ska"
    ],

    "jazz_blues": [
        "jazz", "blues"
    ],

    "acoustic_folk": [
        "acoustic", "folk", "singer-songwriter",
        "guitar", "piano", "bluegrass",
        "country", "honky-tonk"
    ],

    "classical_instrumental": [
        "classical", "opera", "new-age"
    ],

    "world_latin": [
        "latin", "latino", "salsa", "samba", "tango",
        "mpb", "pagode", "forro", "sertanejo", "brazil",
        "spanish", "french", "german", "swedish",
        "indian", "iranian", "turkish", "malay",
        "world-music"
    ],

    "asian_pop_animation": [
        "k-pop", "j-pop", "j-rock", "j-idol",
        "j-dance", "cantopop", "mandopop", "anime"
    ],

    "mood_soft": [
        "chill", "happy", "sad", "romance"
    ]
}


def get_similarity_weighted(vector_a: np.ndarray[float], vector_b: np.ndarray[float]) -> float:
    # Apply the weights to the vectors
    a_weighted = vector_a * WEIGHTS
    b_weighted = vector_b * WEIGHTS
    a_norm = np.linalg.norm(a_weighted)
    b_norm = np.linalg.norm(b_weighted)

    # calling get_similarity with weighted vectors
    return get_similarity(
        a_weighted,
        b_weighted,
        a_norm,
        b_norm
    )

def get_similarity(vector_a: np.ndarray[float], vector_b: np.ndarray[float], vector_a_norm: float, vector_b_norm: float) -> float:
    # Calculate cosine similarity between the two vectors
    return np.dot(vector_a, vector_b) / (vector_a_norm * vector_b_norm)