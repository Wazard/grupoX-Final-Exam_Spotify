import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd
import enum
import time
import pyperclip as clip
import requests
from io import BytesIO

# ================= BACKEND IMPORTS =================
from features.spotify_api           import get_spotify_links_and_images, SpotifyTokenManager
from recommender.model_data         import generate_linear_model_rank, train_linear_models_if_needed
from recommender.light_gbm_new      import generate_lgbm_rank, train_lgbms_if_needed
from recommender.new_cold_start     import generate_cold_start_songs
from recommender.fallback_optimized import generate_fallback_songs
from recommender.ranking            import generate_ranking
from features.song_representation   import vectorize_song
from user.profile                   import UserProfile
from visualization.visualize_plots  import plot_global_genre_radar_from_json


# ======================================================
# GLOBAL CONFIG
# ======================================================

APP_WIDTH = 480
APP_HEIGHT = int(APP_WIDTH / 10 * 16)

PROFILE_PATH = "user/user_profile.json"
DATA_PATH = "data/processed/tracks_processed_normalized.csv"

HOME_BG_PATH = "assets/home_bg.gif"
EXIT_ICON_PATH = "assets/exit_icon.png"

BATCH_SIZE = 10
COLD_START_BATCH_MUL = 2

SHOULD_LOG = True
IMAGE_CACHE = {}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


def log(msg):
    if SHOULD_LOG:
        print(msg)


# ======================================================
# MAIN APP
# ======================================================

class MusicTinderApp(ctk.CTk):

    class Recommender(enum.Enum):
        COLD_START   = 0
        FALLBACK     = 1
        RANKING      = 2
        LINEAR_MODEL = 3
        BOOST_MODEL  = 4

    CARD_W_REF = 360
    CARD_H_REF = int(CARD_W_REF * 1.5)
    ART_SIZE = (250, 250)

    SWIPE_DIST_REF = 420
    SWIPE_STEP_REF = 24

    POPUP_PLOT_W = 720
    POPUP_PLOT_H = 520
    POPUP_USAGE_W = 600
    POPUP_USAGE_H = 420

    # ==================================================
    # INIT
    # ==================================================

    def __init__(self):
        super().__init__()

        self.df = pd.read_csv(DATA_PATH)
        self.df["track_id"] = self.df["track_id"].astype(str).str.strip()

        self.user_profile = UserProfile.load()
        self.token_manager = SpotifyTokenManager()

        self.last_linear_model_train_seen = 0
        self.last_boost_model_train_seen = 0

        self.recommendations = pd.DataFrame()
        self.song_index = 0
        self.current_song = None

        self.scale = 1.0
        self.current_frame = None
        self.resize_handler = None
        self.bg_after_id = None

        self.title("Music Tinder")
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")

        self.bind("<Configure>", self.on_resize)
        self.show_home()

    # ==================================================
    # RESIZE
    # ==================================================

    def update_scale(self):
        self.scale = min(
            self.winfo_width() / APP_WIDTH,
            self.winfo_height() / APP_HEIGHT
        )

    def on_resize(self, event):
        self.update_scale()
        if self.resize_handler:
            self.resize_handler()

    # ==================================================
    # FRAME HANDLING
    # ==================================================

    def clear_frame(self):
        self.stop_background_animation()
        if self.current_frame:
            self.current_frame.destroy()

    # ==================================================
    # HOME SCREEN
    # ==================================================

    def show_home(self):

        self.clear_frame()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(expand=True, fill="both")
        self.current_frame = frame

        self.bg_canvas = tk.Canvas(frame, highlightthickness=0)
        self.bg_canvas.pack(expand=True, fill="both")

        self.load_background()

        ctk.CTkLabel(
            frame,
            text="🎵 Music Tinder",
            font=ctk.CTkFont(size=34, weight="bold")
        ).place(relx=0.5, rely=0.06, anchor="center")

        controls = ctk.CTkFrame(frame, fg_color="transparent", corner_radius=24)
        controls.place(relx=0.03, rely=0.78, anchor="w")

        for text, cmd in [
            ("Start", self.show_tinder),
            ("Music plot", self.open_music_plot),
            ("Usage", self.open_usage),
        ]:
            ctk.CTkButton(
                controls,
                text=text,
                width=200,
                height=40,
                corner_radius=20,
                command=cmd
            ).pack(padx=20, pady=8)

        exit_img = Image.open(EXIT_ICON_PATH).resize((26, 26), Image.LANCZOS)
        self.exit_icon = ImageTk.PhotoImage(exit_img)

        ctk.CTkButton(
            frame,
            text=" Exit",
            image=self.exit_icon,
            compound="left",
            fg_color="#8b0000",
            width=130,
            height=42,
            corner_radius=20,
            command=self.confirm_exit
        ).place(relx=0.97, rely=0.94, anchor="se")

        self.resize_handler = None

    # ==================================================
    # BACKGROUND
    # ==================================================

    def load_background(self):
        img = Image.open(HOME_BG_PATH)
        self.bg_frames = [f.copy() for f in ImageSequence.Iterator(img)]
        self.bg_index = 0
        self.bg_image_id = self.bg_canvas.create_image(0, 0, anchor="nw")
        self.animate_background()

    def zoom_to_fill(self, img, w, h):
        iw, ih = img.size
        scale = max(w / iw, h / ih)
        img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
        left = (img.width - w) // 2
        top = (img.height - h) // 2
        return img.crop((left, top, left + w, top + h))

    def animate_background(self):
        if not self.bg_canvas.winfo_exists():
            return

        frame = self.bg_frames[self.bg_index]
        fitted = self.zoom_to_fill(frame, self.winfo_width(), self.winfo_height())
        self.bg_photo = ImageTk.PhotoImage(fitted)
        self.bg_canvas.itemconfig(self.bg_image_id, image=self.bg_photo)

        self.bg_index = (self.bg_index + 1) % len(self.bg_frames)
        self.bg_after_id = self.after(60, self.animate_background)

    def stop_background_animation(self):
        if self.bg_after_id:
            self.after_cancel(self.bg_after_id)
            self.bg_after_id = None

    # ==================================================
    # RECOMMENDER LOGIC (UNCHANGED)
    # ==================================================

    def choose_recommender(self):
        c = self.user_profile.confidence
        n = len(self.user_profile.seen_song_ids)
        log(f"[RECOMMENDER] confidence={c}, seen={n}")

        if c < 2:
            return self.Recommender.COLD_START
        if c < 4:
            return self.Recommender.FALLBACK
        if c < 8 or n < 200:
            return self.Recommender.RANKING
        if c < 12 or n < 2000:
            return self.Recommender.LINEAR_MODEL
        return self.Recommender.BOOST_MODEL

    def generate_recommendations(self):
        mode = self.choose_recommender()
        start = time.perf_counter()

        log(f"[MODE] mode={mode}")


        if mode == self.Recommender.COLD_START:
            recs = generate_cold_start_songs(self.df, BATCH_SIZE * COLD_START_BATCH_MUL)
        elif mode == self.Recommender.FALLBACK:
            recs = generate_fallback_songs(
                self.df, self.user_profile,
                self.user_profile.seen_song_ids, BATCH_SIZE
            )
        elif mode == self.Recommender.RANKING:
            recs = generate_ranking(self.df, self.user_profile, BATCH_SIZE)
        elif mode == self.Recommender.LINEAR_MODEL:
            self.last_linear_model_train_seen = train_linear_models_if_needed(
                self.df, self.user_profile, self.last_linear_model_train_seen
            )
            recs = generate_linear_model_rank(
                self.df, self.user_profile,
                self.user_profile.seen_song_ids, BATCH_SIZE
            )
        else:
            self.last_boost_model_train_seen = train_lgbms_if_needed(
                self.df, self.user_profile, self.last_boost_model_train_seen
            )
            recs = generate_lgbm_rank(
                self.df, self.user_profile,
                self.user_profile.seen_song_ids, BATCH_SIZE
            )

        token = self.token_manager.get_token()
        self.recommendations = self.attach_spotify_data(recs, token)
        self.song_index = 0

        log(f"[INFO] Generated in {time.perf_counter() - start:.3f}s")

    def attach_spotify_data(self, df, token):
        tmp = df.copy()
        ids = tmp["track_id"].tolist()
        data = get_spotify_links_and_images(ids, token)
        tmp["track_url"] = tmp["track_id"].map(lambda i: data.get(i, {}).get("spotify_url"))
        tmp["album_image"] = tmp["track_id"].map(lambda i: data.get(i, {}).get("image_url"))
        return tmp

    # ==================================================
    # TINDER SCREEN
    # ==================================================

    def show_tinder(self):
        self.clear_frame()
        self.generate_recommendations()

        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both")
        self.current_frame = frame

        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(top, text="←", width=40, command=self.show_home).pack(side="left")
        ctk.CTkLabel(top, text="👤 USER").pack(side="right")

        self.canvas = tk.Canvas(frame, highlightthickness=0, bg="#0f0f0f")
        self.canvas.pack(expand=True, fill="both")

        controls = ctk.CTkFrame(frame)
        controls.pack(pady=10)

        ctk.CTkButton(
            controls, text="❌",
            font=ctk.CTkFont(size=28),
            fg_color="#7a1a1a",
            command=lambda: self.vote(False)
        ).pack(side="left", padx=60)

        ctk.CTkButton(
            controls, text="❤️",
            font=ctk.CTkFont(size=28),
            fg_color="#1a7a4a",
            command=lambda: self.vote(True)
        ).pack(side="right", padx=60)

        self.create_card()
        self.resize_handler = self.resize_tinder
        self.resize_tinder()
        self.load_current_song()

    # ==================================================
    # CARD
    # ==================================================

    def create_card(self):
        self.card = self.canvas.create_rectangle(0, 0, 0, 0, fill="#1f1f1f", outline="")
        self.album_art_id = self.canvas.create_image(0, 0)
        self.card_text = self.canvas.create_text(
            0, 0, fill="white",
            font=("Segoe UI", 18), justify="center"
        )

    def resize_tinder(self):
        if not self.canvas.winfo_exists():
            return

        w, h = self.winfo_width(), self.winfo_height()
        cw = int(self.CARD_W_REF * self.scale)
        ch = int(self.CARD_H_REF * self.scale)

        vertical_offset = int(h * 0.1)
        cx, cy = w // 2, h // 2 - vertical_offset

        self.card_center = (cx, cy)
        self.card_size = (cw, ch)

        self.canvas.coords(
            self.card,
            cx - cw // 2, cy - ch // 2,
            cx + cw // 2, cy + ch // 2
        )

    # ==================================================
    # IMAGE + TEXT
    # ==================================================

    def load_album_image(self, url, max_size):
        if not url:
            return None
        if url in IMAGE_CACHE:
            return IMAGE_CACHE[url]

        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")

        w, h = (self.ART_SIZE[0]*.75, self.ART_SIZE[1] * .75)
        mw, mh = max_size
        scale = min(mw / w, mh / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        IMAGE_CACHE[url] = tk_img
        return tk_img

    def load_current_song(self):
        if self.song_index >= len(self.recommendations):
            self.generate_recommendations()

        song = self.recommendations.iloc[self.song_index]
        self.current_song = song
        log(f"[SONG] {song['track_name']}")

        img = self.load_album_image(song["album_image"], self.ART_SIZE)

        art_x = self.card_center[0]
        art_y = self.card_center[1] - self.card_size[1] // 2 + self.ART_SIZE[1] // 2 + 20

        if img:
            self.canvas.itemconfig(self.album_art_id, image=img)
            self.canvas.coords(self.album_art_id, art_x, art_y)

        text = f"Track name: {song['track_name']}\nArtist: {song['artists']}\nGenre: {song['track_genre']}"
        self.canvas.itemconfig(
            self.card_text,
            text=text,
            width=int(self.card_size[0] * 0.9)
        )

        self.canvas.coords(
            self.card_text,
            self.card_center[0],
            art_y + self.ART_SIZE[1] // 2 + 60
        )

        if song.get("track_url"):
            clip.copy(song["track_url"])

    # ==================================================
    # VOTING + SWIPE
    # ==================================================

    def vote(self, liked: bool):
        song = self.current_song
        vector, track_id = vectorize_song(song, include_id=True)

        if liked:
            self.user_profile.like(vector.tolist(), track_id, song["track_genre"])
        else:
            self.user_profile.dislike(vector.tolist(), track_id, song["track_genre"])

        self.user_profile.save()
        self.animate_swipe(liked)

    def animate_swipe(self, is_like):
        dx = int(self.SWIPE_STEP_REF * self.scale)
        swipe_dist = self.SWIPE_DIST_REF
        if not is_like:
            dx *= -1
            swipe_dist += self.card_size[0]/2

        def step():
            for item in (self.card, self.album_art_id, self.card_text):
                self.canvas.move(item, dx, 0)

            x1, _, _, _ = self.canvas.coords(self.card)
            if abs(x1 - self.card_center[0]) < swipe_dist:
                self.after(16, step)
            else:
                self.reset_card()
                self.song_index += 1
                self.load_current_song()

        step()

    def reset_card(self):
        cx, cy = self.card_center
        cw, ch = self.card_size

        self.canvas.coords(
            self.card,
            cx - cw // 2, cy - ch // 2,
            cx + cw // 2, cy + ch // 2
        )

    # ==================================================
    # POPUPS
    # ==================================================

    def open_music_plot(self):
        popup = ctk.CTkToplevel(self)
        popup.geometry("720x520")
        fig, _ = plot_global_genre_radar_from_json(PROFILE_PATH)
        canvas = FigureCanvasTkAgg(fig, popup)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def open_usage(self):
        popup = ctk.CTkToplevel(self)
        popup.geometry("600x420")
        ctk.CTkLabel(
            popup,
            text=("Lorem ipsum dolor sit amet.\n\n" * 15),
            wraplength=560
        ).pack(padx=20, pady=20)

    def confirm_exit(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.destroy()


if __name__ == "__main__":
    MusicTinderApp().mainloop()
