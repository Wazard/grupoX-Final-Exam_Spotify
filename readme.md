# MusicTinder 🎵

A terminal-based, offline music recommendation engine that utilizes vector-space modeling to find songs based on their "sonic DNA." Unlike social-based algorithms, **MusicTinder** focuses purely on audio features and implicit user feedback to navigate a local dataset.

---

## 🎯 Core Concept

The system operates on the principle that **a user is the mathematical average of the songs they like.** By representing songs as vectors of audio features, the engine calculates the distance between your "User Profile" and the rest of the library to find your next favorite track.

## 🏗️ Data Architecture

The dataset is categorized into four functional layers to balance similarity, ranking, and display:

### A. Core Similarity (The "Sound")
These features are the *only* inputs used for similarity calculations.
* **Energy, Danceability, & Valence:** The mood and movement.
* **Acousticness, Instrumentalness, & Liveness:** The texture of the recording.
* **Tempo, Loudness, & Speechiness:** Technical audio properties.

### B. Ranking Modifiers
Applied as weights *after* the similarity calculation to refine the final recommendation list.
* **Popularity:** Boosts or penalizes tracks based on global reach.
* **Explicit:** Filters content.
* **Duration (ms):** Ensures recommendations meet length criteria.

### C. Identity (Display & Grouping)
Used for the CLI output and for applying diversity constraints.
* `track_name`, `artist`, `album_name`, `track_id`, `track_genre`.

### D. Discrete Metadata
Categorical values normalized using `scikit-learn` (StandardScaler) before processing.
* `key`, `mode`, `time_signature`.

---

## 🧠 Recommendation Logic

### 1. The Cold Start (Default List)
When no user history exists, MusicTinder generates a **Default Discovery** list:
* **High Popularity:** Surfaces generally well-liked tracks.
* **Genre Diversity:** Ensures the initial tracks cover a broad spectrum to probe user interests.

### 2. Implicit Feedback Loop
The engine relies on a "Like/Dislike" system to evolve the user profile:
* **Like:** The song's feature vector is averaged into the **User Profile**.
* **Dislike:** The song is ignored or used to downweight similar future suggestions.

### 3. Diversity Constraint
To prevent repetitive results, the algorithm applies a grouping constraint on `album_name` or `artist`, ensuring the top results represent a diverse range of creators.

---

## 🛠️ Tech Stack

* **Interface:** CLI (No GUI)
* **Environment:** Fully Offline
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Normalization & Distance metrics)

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Wazard/grupoX-Final-Exam_Spotify.git](https://github.com/Wazard/grupoX-Final-Exam_Spotify.git)
