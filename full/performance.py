"""
Tracking e grafici delle prestazioni.

Idea: se il modello impara, nel tempo aumenta:
- cumulative like rate
- rolling like rate
- calibrazione tra p_pred e outcome (grossolana)
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

from config import ROLLING_WINDOW


@dataclass
class Tracker:
    rows: list

    def __init__(self):
        self.rows = []

    def log(self, t: int, track_id: str, cluster: int, p_pred: float, reward: int):
        self.rows.append({
            "t": int(t),
            "track_id": str(track_id),
            "cluster_idx": int(cluster),
            "p_pred": float(p_pred),
            "reward": int(reward),
        })

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if len(df) == 0:
            return df
        df["cum_like_rate"] = df["reward"].expanding().mean()
        df["rolling_like_rate"] = df["reward"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        return df

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        df = self.to_frame()
        csv_path = os.path.join(out_dir, "interaction_log.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def plot_all(self, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        df = self.to_frame()
        if len(df) == 0:
            return {}

        paths = {}

        # 1) Cumulative like rate
        plt.figure()
        plt.plot(df["t"], df["cum_like_rate"])
        plt.xlabel("t (turni)")
        plt.ylabel("Cumulative like rate")
        plt.title("Apprendimento nel tempo (cumulativo)")
        p1 = os.path.join(out_dir, "cum_like_rate.png")
        plt.savefig(p1, bbox_inches="tight")
        plt.close()
        paths["cum_like_rate"] = p1

        # 2) Rolling like rate
        plt.figure()
        plt.plot(df["t"], df["rolling_like_rate"])
        plt.xlabel("t (turni)")
        plt.ylabel(f"Rolling like rate (window={ROLLING_WINDOW})")
        plt.title("Apprendimento nel tempo (rolling)")
        p2 = os.path.join(out_dir, "rolling_like_rate.png")
        plt.savefig(p2, bbox_inches="tight")
        plt.close()
        paths["rolling_like_rate"] = p2

        # 3) Pred prob vs outcome (binned reliability)
        plt.figure()
        bins = np.linspace(0, 1, 11)
        df["bin"] = pd.cut(df["p_pred"], bins=bins, include_lowest=True)
        calib = df.groupby("bin", observed=False).agg(
            p_mean=("p_pred", "mean"),
            y_mean=("reward", "mean"),
            n=("reward", "count"),
        ).reset_index(drop=True)

        plt.plot(calib["p_mean"], calib["y_mean"], marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Probabilità predetta (media bin)")
        plt.ylabel("Frazione like (media bin)")
        plt.title("Calibrazione grezza (binning)")
        p3 = os.path.join(out_dir, "calibration.png")
        plt.savefig(p3, bbox_inches="tight")
        plt.close()
        paths["calibration"] = p3

        return paths
