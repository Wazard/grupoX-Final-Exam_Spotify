import numpy as np

SIMILARITY_FEATURES = ['danceability','energy','valence','tempo','loudness','acousticness','instrumentalness','liveness','speechiness']
RANKING_FEATURES = ['popularity','explicit','duration_ms']
IDENTITY_FEATURES = ['track_name','artist','album_name','track_id']

def similarity(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))