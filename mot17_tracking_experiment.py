# mot17_tracking_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import motmetrics as mm

from utils.io import load_mot_txt, detections_by_frame
from trackers.ocsort import *  # utilise le tracker que tu veux
from trackers.bytetrack import *
from trackers.customtrackers import *



def run_sequence(det_path: Path, tracker_cls) -> List[tuple]:
    det_df = load_mot_txt(det_path)
    det_by_frame = detections_by_frame(det_df)
    tracker = tracker_cls()
    results = []
    max_frame = int(det_df["frame"].max())
    for f in range(1, max_frame + 1):
        dets = det_by_frame.get(f, [])
        tracker.update(dets)
        results.extend(tracker.tracks_as_mot(f))
    return results


def evaluate(seq_name: str, gt_path: Path, res_rows: List[tuple]) -> mm.MOTAccumulator:
    gt_df = load_mot_txt(gt_path)
    res_df = pd.DataFrame(res_rows, columns=gt_df.columns)

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        res_frame = res_df[res_df["frame"] == frame_id]

        gt_ids = gt_frame["id"].values
        gt_boxes = gt_frame[["bb_left", "bb_top", "bb_width", "bb_height"]].values

        res_ids = res_frame["id"].values
        res_boxes = res_frame[["bb_left", "bb_top", "bb_width", "bb_height"]].values

        if len(gt_boxes) == 0 or len(res_boxes) == 0:
            dist_matrix = np.empty((len(gt_boxes), len(res_boxes)))
            dist_matrix[:] = np.nan
        else:
            dist_matrix = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)

        acc.update(gt_ids, res_ids, dist_matrix)

    acc.events.loc[:, "Sequence"] = seq_name
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", type=Path, required=True, help="PATH/MOT17/train")
    ap.add_argument("--output_dir", type=Path, default=Path("results"))
    ap.add_argument("--sequences", nargs="*", default=None, help="e.g. 02 05 10")
    ap.add_argument("--tracker", type=str, default="ocsort", help="ocsort | bytetrack | simple")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    
    tracker_dict = {
        "ocsort": OC_SORT,
        "bytetrack": ByteTrack, 
        "simple": CustomTracker,
    }
    tracker_cls = tracker_dict[args.tracker.lower()]

    mm.lap.default_metric = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    accs = []

    seq_list = args.sequences or sorted(p.name for p in Path(args.mot_root).glob("MOT17-*"))
    for seq in tqdm(seq_list, desc="Sequences"):
        seq_dir = Path(args.mot_root) / f"MOT17-{seq}-DPM"
        det_path = seq_dir / "det/det.txt"
        gt_path = seq_dir / "gt/gt.txt"
        res_rows = run_sequence(det_path, tracker_cls)

        res_file = args.output_dir / f"MOT17-{seq}-{args.tracker}.txt"
        np.savetxt(res_file, res_rows, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d")

        acc = evaluate(seq, gt_path, res_rows)
        accs.append(acc)
        seq_name = f"MOT17-{seq}-FRCNN"
        summary = mh.compute(acc, name=seq_name)

        mota = summary.at[seq_name, "mota"] * 100
        idf1 = summary.at[seq_name, "idf1"] * 100 if "idf1" in summary.columns else float("nan")
        idp = summary.at[seq_name, "idp"] * 100 if "idp" in summary.columns else float("nan")
        idr = summary.at[seq_name, "idr"] * 100 if "idr" in summary.columns else float("nan")
        num_switches = summary.at[seq_name, "num_switches"] if "num_switches" in summary.columns else int("nan")
        mostly_tracked = summary.at[seq_name, "mostly_tracked"] if "mostly_tracked" in summary.columns else int("nan")
        num_fragmentations = summary.at[seq_name, "num_fragmentations"] if "num_fragmentations" in summary.columns else int("nan")

        print(f"✅ {seq_name} → MOTA: {mota:.2f}%, IDF1: {idf1:.2f}%")
        print(f"✅ {seq_name} → idp: {idp:.2f}%, idr: {idr:.2f}%")
        print(f"✅ {seq_name} → num_switches: {num_switches}, mostly_tracked: {mostly_tracked}")
        print(f"✅ {seq_name} → num_fragmentations: {num_fragmentations}")

    summary = mh.compute_many(accs, names=[a.events["Sequence"].iloc[0] for a in accs])
    str_summary = mm.io.render_summary(summary, namemap=lambda s: s)
    (args.output_dir / "summary.txt").write_text(str_summary)


if __name__ == "__main__":
    main()
