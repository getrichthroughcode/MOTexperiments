from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd


def load_mot_txt(path: Path) -> pd.DataFrame:
    cols = [
        "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf",
        "x", "y", "z",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def detections_by_frame(det_df: pd.DataFrame) -> Dict[int, List[dict]]:
    """
    Transforme un DataFrame MOT en dictionnaire indexé par frame.
    Chaque détection est un dictionnaire contenant :
        - bbox: (x_center, y_center, w, h)
        - measurement: (x_center, y_center, aspect_ratio, height)
        - label: "person"
        - score: float
    """
    out: Dict[int, List[dict]] = {}
    for frame, fdf in det_df.groupby("frame"):
        dets: List[dict] = []
        for _, r in fdf.iterrows():
            x1, y1, w, h = r["bb_left"], r["bb_top"], r["bb_width"], r["bb_height"]
            x_center, y_center = x1 + w / 2.0, y1 + h / 2.0
            dets.append(
                {
                    "bbox": (x_center, y_center, w, h),
                    "measurement": np.array([x_center, y_center, w / h, h], dtype=float),
                    "label": "person",
                    "score": r["conf"],
                }
            )
        out[int(frame)] = dets
    return out
