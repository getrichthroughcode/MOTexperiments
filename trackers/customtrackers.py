from trackers.base import BaseTracker
from models.kalman import KalmanFilter, chi2inv95
import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment


class SimpleTrack:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, score: float):
        self.mean = mean
        self.cov = cov
        self.score = score
        self.history: List[Tuple[int, int, int, int]] = []
        self.missed = 0


class CustomTracker(BaseTracker):
    def __init__(self):
        self.kf = KalmanFilter()
        self.tracks: Dict[int, SimpleTrack] = {}
        self.next_id = 1

    def update(self, detections: List[dict]):
        detections = [d for d in detections if d["score"] >= 0.3]
        self._predict()
        matches, unmatched_trk, unmatched_det = self._associate(detections)

        if unmatched_trk and unmatched_det:
            fallback_matches, unmatched_trk, unmatched_det = self._associate_iou(unmatched_trk, unmatched_det, detections)
            matches.update(fallback_matches)

        self._update_matched(detections, matches)
        self._init_new_tracks(detections, unmatched_det)
        self._remove_lost(unmatched_trk)

    def tracks_as_mot(self, frame_idx: int):
        rows = []
        for tid, trk in self.tracks.items():
            x1, y1, w, h = self._state_to_tlwh(trk.mean)
            rows.append((frame_idx, tid, x1, y1, w, h, trk.score, -1., -1., -1.))
        return rows

    def _predict(self):
        for trk in self.tracks.values():
            trk.mean, trk.cov = self.kf.prediction(trk.mean, trk.cov)
            trk.missed += 1

    def _associate(self, detections):
        if not self.tracks or not detections:
            return {}, set(self.tracks.keys()), set(range(len(detections)))

        trk_ids = list(self.tracks.keys())
        cost = np.zeros((len(trk_ids), len(detections)))

        for i, tid in enumerate(trk_ids):
            meas = np.asarray([d["measurement"] for d in detections])
            cost[i] = self.kf.gating_distance(self.tracks[tid].mean, self.tracks[tid].cov, meas, False, "maha")

        cost[~np.isfinite(cost)] = 1e5

        r, c = linear_sum_assignment(cost)

        matches, un_trk, un_det = {}, set(trk_ids), set(range(len(detections)))
        for i, j in zip(r, c):
            if cost[i, j] < chi2inv95[4]:
                matches[trk_ids[i]] = j
                un_trk.discard(trk_ids[i])
                un_det.discard(j)

        return matches, un_trk, un_det

    def _associate_iou(self, un_trk, un_det, detections):
        trk_ids = list(un_trk)
        det_ids = list(un_det)
        cost_matrix = np.ones((len(trk_ids), len(det_ids)))

        for i, tid in enumerate(trk_ids):
            trk_box = self._state_to_tlwh(self.tracks[tid].mean)
            for j, did in enumerate(det_ids):
                det_box = self._tlwh_from_detection(detections[did])
                cost_matrix[i, j] = 1 - self._iou(trk_box, det_box)

        cost_matrix[~np.isfinite(cost_matrix)] = 1e5
        row, col = linear_sum_assignment(cost_matrix)

        matches = {}
        unmatched_trk = set(trk_ids)
        unmatched_det = set(det_ids)

        for i, j in zip(row, col):
            if cost_matrix[i, j] < 0.7:
                matches[trk_ids[i]] = det_ids[j]
                unmatched_trk.discard(trk_ids[i])
                unmatched_det.discard(det_ids[j])

        return matches, unmatched_trk, unmatched_det

    def _tlwh_from_detection(self, detection):
        x_c, y_c, w, h = detection["bbox"]
        x1, y1 = x_c - w / 2.0, y_c - h / 2.0
        return x1, y1, w, h

    def _state_to_tlwh(self, mean: np.ndarray) -> Tuple[float, float, float, float]:
        x, y, a, h = mean[:4]
        w = a * h
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        return x1, y1, w, h

    def _iou(self, boxA, boxB):
        ax1, ay1, aw, ah = boxA
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = boxB
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        boxA_area = aw * ah
        boxB_area = bw * bh

        union = boxA_area + boxB_area - inter_area
        return inter_area / union if union > 0 else 0

    def _update_matched(self, detections, matches):
        for tid, d_idx in matches.items():
            det = detections[d_idx]
            mean, cov = self.kf.update(self.tracks[tid].mean, self.tracks[tid].cov, det["measurement"])
            self.tracks[tid].mean = mean
            self.tracks[tid].cov = cov
            self.tracks[tid].score = det["score"]
            tlwh = self._state_to_tlwh(mean)
            self.tracks[tid].history.append(tlwh)
            self.tracks[tid].missed = 0

    def _init_new_tracks(self, detections, un_det):
        for d_idx in un_det:
            det = detections[d_idx]
            mean, cov = self.kf.initialiser(det["measurement"])
            self.tracks[self.next_id] = SimpleTrack(mean, cov, det["score"])
            self.next_id += 1

    def _remove_lost(self, un_trk):
        for tid in list(un_trk):
            if self.tracks[tid].missed > 15:
                del self.tracks[tid]
