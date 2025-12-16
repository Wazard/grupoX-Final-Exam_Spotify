from evaluation.metrics import SIMILARITY_FEATURES
global V_LENGTH
V_LENGTH = 18

def vectorize_song(song_row) -> list[float]:

    vector:list[float] = []

    for feature in SIMILARITY_FEATURES:
        vector.append(song_row[feature])

    return vector