import numpy as np

SIMILARITY_FEATURES = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
FEATURE_WEIGHTS = {
    'danceability': 1.0, 'energy': 1.2, 'valence': 0.9, 'tempo': 1.1, 
    'loudness': 1.0, 'acousticness': 0.8, 'instrumentalness': 1.3, 
    'liveness': 0.9, 'speechiness': 1.0
}
WEIGHTS = np.array([FEATURE_WEIGHTS[f] for f in SIMILARITY_FEATURES])

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