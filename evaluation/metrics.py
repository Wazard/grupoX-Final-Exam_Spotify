import numpy as np

SIMILARITY_FEATURES = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
FEATURE_WEIGHTS = {
    'danceability': 1.0, 'energy': 1.2, 'valence': 0.9, 'tempo': 1.1, 
    'loudness': 1.0, 'acousticness': 0.8, 'instrumentalness': 1.3, 
    'liveness': 0.9, 'speechiness': 1.0
}

def get_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    # Convert the vectors to numpy arrays
    a = np.array(vector_a)
    b = np.array(vector_b)

    # Apply the weights to the vectors
    weights = np.array([FEATURE_WEIGHTS[f] for f in SIMILARITY_FEATURES])
    a_weighted = a * weights
    b_weighted = b * weights

    # Calculate cosine similarity between the weighted vectors
    similarity = np.dot(a_weighted, b_weighted) / (np.linalg.norm(a_weighted) * np.linalg.norm(b_weighted))
    
    return similarity
