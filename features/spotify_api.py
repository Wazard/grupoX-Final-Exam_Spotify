
import requests
import base64
import time
import os

SPOTIFY_TRACKS_ENDPOINT = "https://api.spotify.com/v1/tracks"


def get_spotify_links_and_images(
    track_ids: list[str],
    access_token: str
) -> dict[str, dict[str, str]]:
    """
    Given a list of Spotify track IDs, return a mapping:
    track_id -> {
        "spotify_url": str,
        "image_url": str | None
    }
    """

    results = {}

    # Spotify allows max 50 IDs per request
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i + 50]

        params = {
            "ids": ",".join(batch)
        }

        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        r = requests.get(
            SPOTIFY_TRACKS_ENDPOINT,
            headers=headers,
            params=params,
            timeout=10
        )
        r.raise_for_status()

        tracks = r.json().get("tracks", [])

        for track in tracks:
            if track is None:
                continue  # happens if ID is invalid

            track_id = track["id"]

            spotify_url = track["external_urls"]["spotify"]

            images = track["album"].get("images", [])
            image_url = images[-1]["url"] if images else None

            results[track_id] = {
                "spotify_url": spotify_url,
                "image_url": image_url
            }

    return results

class SpotifyTokenManager:
    def __init__(self):
        self.token = None
        self.expires_at = 0

    def get_token(self):
        if time.time() >= self.expires_at:
            self.token, expires_in = self._request_token()
            self.expires_at = time.time() + expires_in - 60
        return self.token

    def _request_token(self):
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {"grant_type": "client_credentials"}

        r = requests.post(
            "https://accounts.spotify.com/api/token",
            headers=headers,
            data=data,
            timeout=10
        )
        r.raise_for_status()

        payload = r.json()
        return payload["access_token"], payload["expires_in"]