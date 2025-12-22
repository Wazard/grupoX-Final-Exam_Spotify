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
    "Pop mainstream": [
        "pop", "indie-pop", "power-pop", "british"
    ],

    "Dance/EDM": [
        "dance", "edm", "club", "party", "disco", "synth-pop"
    ],

    "soundtrack_family": [
        "pop-film", "disney", "show-tunes", "children"
    ],

    "House/Melodic": [
        "house", "deep-house", "progressive-house", "chicago-house", "garage"
    ],

    "Techno/Industrial": [
        "techno", "minimal-techno", "detroit-techno", "industrial"
    ],

    "Bass breaks": [
        "drum-and-bass", "dubstep", "breakbeat", "dub", "electro", "idm"
    ],

    "Rock classic/alternative": [
        "rock", "alt-rock", "alternative", "hard-rock",
        "rock-n-roll", "rockabilly", "psych-rock", "grunge"
    ],

    "Indie/British": [
        "indie", "british", "j-dance"
    ],

    "Metal": [
        "metal", "heavy-metal"
    ],

    "Extreme metal": [
        "death-metal", "black-metal", "grindcore",
        "hardcore", "metalcore"
    ],

    "Punk/Emo/Goth": [
        "punk", "punk-rock", "emo", "goth"
    ],

    "HipHop/RNB": [
        "hip-hop", "r-n-b", "soul", "funk", "groove"
    ],

    "Ragge/Ska": [
        "reggae", "reggaeton", "dancehall", "ska"
    ],

    "Jazz/Blues": [
        "jazz", "blues"
    ],

    "Folk/Acoustic": [
        "acoustic", "folk", "singer-songwriter",
        "guitar", "piano", "bluegrass", "j-dance"
    ],

    "Country": [
        "country", "honky-tonk"
    ],

    "Classical/Opera": [
        "classical", "opera"
    ],

    "New age": [
        "new-age"
    ],

    "Latin dance": [
        "latin", "latino", "salsa", "samba", "tango"
    ],

    "Brazilian": [
        "mpb", "pagode", "forro", "sertanejo", "brazil"
    ],

    "Non-western": [
        "indian", "iranian", "turkish", "malay", "world-music"
    ],

    "European Pop": [
        "spanish", "french", "german", "swedish"
    ],

    "Asian Pop": [
        "k-pop", "j-pop", "j-rock", "j-idol", "cantopop", "mandopop", "anime"
    ],

    "Mood soft": [
        "chill", "sad", "romance"
    ]
}

EPS = 1e-12  # numerical safety

def get_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    vector_a_norm: float,
    vector_b_norm: float,
) -> float:
    """
    Safe cosine similarity.
    Returns 0.0 if similarity is undefined.
    """

    denom = vector_a_norm * vector_b_norm
    if denom < EPS:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denom)


def get_similarity_weighted(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Weighted cosine similarity with zero-norm protection.
    """

    a_weighted = vector_a * WEIGHTS
    b_weighted = vector_b * WEIGHTS

    a_norm = np.linalg.norm(a_weighted)
    b_norm = np.linalg.norm(b_weighted)

    if a_norm < EPS or b_norm < EPS:
        return 0.0

    return float(np.dot(a_weighted, b_weighted) / (a_norm * b_norm))
