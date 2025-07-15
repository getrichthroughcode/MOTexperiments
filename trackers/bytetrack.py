from typing import List, Tuple, Dict
import numpy as np
from trackers.base import BaseTracker
from models.kalman import KalmanFilter, chi2inv95


class Track:
    def __init__(self, mean, cov, score):
        self.mean = mean
        self.cov = cov
        self.score = score
        self.missed = 0
        self.id = None
        self.history = []


class ByteTrack(BaseTracker):
    def __init__(self, score_thresh=0.6, max_age=30):
        self.kf = KalmanFilter()
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.score_thresh = score_thresh
        self.max_age = max_age

    def update(self, detections: List[dict]):
        # 1. Split detections into high and low confidence
        D_high, D_low = [], []
        for d in detections:
            if d["score"] > self.score_thresh:
                D_high.append(d)
            else:
                D_low.append(d)

        # 2. Predict all tracks
        for trk in self.tracks.values():
            trk.mean, trk.cov = self.kf.prediction(trk.mean, trk.cov)
            trk.missed += 1

        # 3. First association with D_high
        matches, unmatched_tracks, unmatched_high = self._associate(list(self.tracks.items()), D_high)
        for tid, d_idx in matches.items():
            self._update_track(tid, D_high[d_idx])

        # 4. Second association with D_low
        remaining_tracks = [(tid, self.tracks[tid]) for tid in unmatched_tracks]
        matches2, final_unmatched_tracks, _ = self._associate(remaining_tracks, D_low)
        for tid, d_idx in matches2.items():
            self._update_track(tid, D_low[d_idx])

        # 5. Delete unmatched tracks
        for tid in final_unmatched_tracks:
            if self.tracks[tid].missed > self.max_age:
                del self.tracks[tid]

        # 6. Init new tracks from unmatched high detections
        matched_high = set(matches.values())
        for i, det in enumerate(D_high):
            if i not in matched_high:
                mean, cov = self.kf.initialiser(det["measurement"])
                trk = Track(mean, cov, det["score"])
                trk.id = self.next_id
                self.tracks[self.next_id] = trk
                self.next_id += 1

    def _update_track(self, tid: int, det: dict):
        trk = self.tracks[tid]
        trk.mean, trk.cov = self.kf.update(trk.mean, trk.cov, det["measurement"])
        trk.score = det["score"]
        trk.missed = 0
        trk.history.append(self._state_to_tlwh(trk.mean))

    def _associate(self, tracks: List[Tuple[int, Track]], detections: List[dict]):
        if not tracks or not detections:
            return {}, set([tid for tid, _ in tracks]), set(range(len(detections)))

        tid_list = [tid for tid, _ in tracks]
        cost = np.zeros((len(tracks), len(detections)))

        for i, (tid, trk) in enumerate(tracks):
            meas = np.asarray([d["measurement"] for d in detections])
            cost[i] = self.kf.gating_distance(trk.mean, trk.cov, meas, False, "maha")

        cost[~np.isfinite(cost)] = 1e5
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)

        matches, un_trk, un_det = {}, set(tid_list), set(range(len(detections)))
        for i, j in zip(row, col):
            if cost[i, j] < chi2inv95[4]:
                matches[tid_list[i]] = j
                un_trk.discard(tid_list[i])
                un_det.discard(j)

        return matches, un_trk, un_det

    def _state_to_tlwh(self, mean):
        x, y, a, h = mean[:4]
        w = a * h
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        return x1, y1, w, h

    def tracks_as_mot(self, frame_idx: int):
        rows = []
        for tid, trk in self.tracks.items():
            x1, y1, w, h = self._state_to_tlwh(trk.mean)
            rows.append((frame_idx, tid, x1, y1, w, h, trk.score, -1., -1., -1.))
        return rows
