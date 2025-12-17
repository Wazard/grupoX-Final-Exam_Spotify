from evaluation.metrics import SIMILARITY_FEATURES
import pandas as pd
import numpy as np

def vectorize_song(song_row, include_id: bool = False) -> tuple[np.ndarray, str] | np.ndarray:
    '''
    takes a song, returns a vectorized form + its id
    
    :param song_row: Description
    :return: Description
    :rtype: tuple[np.ndarray, str] | np.ndarray
    '''
    # Pre-allocate numpy array
    vector = song_row[SIMILARITY_FEATURES].to_numpy(dtype=np.float32)
    
    if include_id:
        return vector, song_row['track_id']
    return vector

def vectorize_songs_batch(df, include_ids: bool = False):
    """
    Vectorize multiple songs at once (much faster!).
    
    Args:
        df: DataFrame with song data
        include_ids: Whether to include track IDs
    
    Returns:
        Either array of vectors or tuple of (vectors, ids)
    """
    # Direct column selection - vectorized operation
    vectors = df[SIMILARITY_FEATURES].to_numpy(dtype=np.float32)
    
    if include_ids:
        ids = df['track_id'].values
        return vectors, ids
    return vectors