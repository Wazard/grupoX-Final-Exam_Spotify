Single user

No GUI

Offline dataset only

Implicit feedback only (like / dislike)

Goal: suggest relevant and diverse songs

#A core similarities:

-danceability
-energy
-valence
-tempo
-loudness
-acousticness
-instrumentalness
-liveness
-speechiness

how a song "sounds"
These are the only inputs to similarity calculations.

#B Ranking modifiers:

-popularity
-explicit
-duration_ms

criteria for song rankings applied after similarity calculation.

#C Identity:

-track_name
-artist
-album_name
-track_id
-track_genre

displayable columns only (album could be used as a grouping or diversity constraint)


#D Discrete metadata:

-key
-mode
-time_signature

normalize dataset (use scikit-learn)


Create a “default recommendation list” when no user-data is present

-Based on popularity
-With artist and genre variety


User profile:
 `a user is the average of the songs they like`
-User likes a song -> update profile
-User dislikes a song -> ignore or downweight (might add "indifferent" choice)