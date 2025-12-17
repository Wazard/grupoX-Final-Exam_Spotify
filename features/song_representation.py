from evaluation.metrics import SIMILARITY_FEATURES

def vectorize_song(song_row, include_id: bool = False) -> tuple[list[float],str] | list[float]:
    '''
    takes a song, returns a vectorized form + its id
    
    :param song_row: Description
    :return: Description
    :rtype: tuple[list[float], str]
    '''
    vector:list[float] = []

    for feature in SIMILARITY_FEATURES:
        vector.append(song_row[feature])

    if include_id:
        song_id = song_row['track_id']
        return vector,song_id
    return vector