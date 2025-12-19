from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json


# ==========================
# CONFIG
# ==========================

N_TOP_PROFILES = 5          # how many taste profiles to plot
N_TOP_GENRES = 12           # how many genres in global radar
MIN_CONFIDENCE = 0.5        # minimum confidence for genre aggregation

def plot_active_profile_vectors_from_json(
    profile_json_path: str,
    feature_names: list[str],
):
    with open(profile_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        sorted_profiles = sorted(
            data.get("taste_profiles", []), 
            key=lambda p: p.get('confidence', 0), 
            reverse=True
        )

    for p in sorted_profiles[:N_TOP_PROFILES]:
        vector = np.array(p["vector"], dtype=float)

        # === ACTIVE PROFILE RULE ===
        if p.get("liked_count", 0) == 0:
            continue
        if vector.sum() == 0:
            continue
        if len(vector) != len(feature_names):
            continue

        plot_profile_vector_radar(
            profile_name=f"{p['cluster_name']} (conf={p['confidence']:.2f})",
            vector=vector,
            feature_names=feature_names,
        )


def plot_profile_vector_radar(
    profile_name: str,
    vector: np.ndarray,
    feature_names: list[str],
):
    if len(vector) != len(feature_names):
        raise ValueError("vector and feature_names must have same length")

    v = np.abs(vector.astype(float))
    if v.sum() == 0:
        return

    # Normalize for shape comparison
    v = v / v.max()

    angles = np.linspace(0, 2 * np.pi, len(v), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    v = np.concatenate([v, [v[0]]])

    _, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, v, linewidth=2)
    ax.fill(angles, v, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels([])

    ax.set_title(f"Taste Fingerprint — {profile_name}")
    plt.show()


# ==========================
# GENRE-SPACE RADAR
# ==========================

def build_global_genre_vector(
    profile_json_path: str,
):
    """
    Aggregate genre preferences across active profiles.
    Weight = genre_count × profile_confidence
    """

    with open(profile_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    genre_weights = defaultdict(float)

    for p in data.get("taste_profiles", []):
        if p.get("liked_count", 0) == 0:
            continue
        if p.get("confidence", 0.0) < MIN_CONFIDENCE:
            continue

        conf = p["confidence"]

        for genre, count in p.get("genre_counts", {}).items():
            genre_weights[genre] += count * conf

    return dict(genre_weights)


def plot_global_genre_radar(
    genre_weights: dict[str, float],
):
    """
    Radar plot showing the user's overall genre preferences.
    """

    if not genre_weights:
        return

    items = sorted(
        genre_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )[:N_TOP_GENRES]

    labels = [g for g, _ in items]
    values = np.array([v for _, v in items], dtype=float)

    values = values / values.max()

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    values = np.concatenate([values, [values[0]]])

    _, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, color="darkgreen")
    ax.fill(angles, values, alpha=0.3, color="green")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    ax.set_title("User Genre Preference Fingerprint")
    plt.show()


def plot_global_genre_radar_from_json(
    profile_json_path: str,
):
    genre_vector = build_global_genre_vector(profile_json_path)
    plot_global_genre_radar(genre_vector)


# ==========================
# END OF FILE
# ==========================
