from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from trackers.base import BaseTracker
from models.kalman import KalmanFilter, chi2inv95
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB):
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


def tlwh_from_state(mean):
    x, y, a, h = mean[:4]
    w = a * h
    return x - w / 2, y - h / 2, w, h


def tlwh_from_det(det):
    x, y, w, h = det["bbox"]
    return x - w / 2, y - h / 2, w, h


class OCTrack:
    def __init__(self, mean, cov, score, det):
        self.mean = mean
        self.cov = cov
        self.score = score
        self.history = [det]
        self.missed = 0
        self.tracked = False


class OC_SORT(BaseTracker):
    def __init__(self, max_age=30, lambda_vel=0.2):
        self.kf = KalmanFilter()
        self.tracks: Dict[int, OCTrack] = {}
        self.next_id = 1
        self.max_age = max_age
        self.lambda_vel = lambda_vel

    def update(self, detections: List[dict]):
        # Prédiction Kalman
        for trk in self.tracks.values():
            trk.mean, trk.cov = self.kf.prediction(trk.mean, trk.cov)
            trk.missed += 1

        # Association primaire (IoU + vitesse)
        matches, unmatched_trks, unmatched_dets = self._associate(detections)

        # OCR step pour les tracks non appariés
        matches_ocr, unmatched_trks, unmatched_dets = self._ocr_associate(detections, unmatched_trks, unmatched_dets)
        matches.update(matches_ocr)

        # Mise à jour
        for tid, d_idx in matches.items():
            det = detections[d_idx]
            mean, cov = self.kf.update(self.tracks[tid].mean, self.tracks[tid].cov, det["measurement"])
            self.tracks[tid].mean = mean
            self.tracks[tid].cov = cov
            self.tracks[tid].score = det["score"]
            self.tracks[tid].history.append(det)
            self.tracks[tid].missed = 0
            self.tracks[tid].tracked = True

        # Initialiser nouveaux
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            mean, cov = self.kf.initialiser(det["measurement"])
            self.tracks[self.next_id] = OCTrack(mean, cov, det["score"], det)
            self.tracks[self.next_id].tracked = False
            self.next_id += 1

        # Supprimer les anciens
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_age:
                del self.tracks[tid]

    def tracks_as_mot(self, frame_idx: int):
        rows = []
        for tid, trk in self.tracks.items():
            if trk.tracked:
                x1, y1, w, h = tlwh_from_state(trk.mean)
                rows.append((frame_idx, tid, x1, y1, w, h, trk.score, -1., -1., -1.))
        return rows

    def _associate(self, detections):
        if not self.tracks or not detections:
            return {}, set(self.tracks.keys()), set(range(len(detections)))

        trk_ids = list(self.tracks.keys())
        cost = np.zeros((len(trk_ids), len(detections)))

        for i, tid in enumerate(trk_ids):
            trk_box = tlwh_from_state(self.tracks[tid].mean)
            meas = np.asarray([d["measurement"] for d in detections])
            for j, det in enumerate(detections):
                det_box = tlwh_from_det(det)
                iou_cost = 1 - iou(trk_box, det_box)
                vel_cost = self.kf.gating_distance(self.tracks[tid].mean, self.tracks[tid].cov, meas[[j]], False, "maha")[0] / chi2inv95[4]
                cost[i, j] = iou_cost + self.lambda_vel * vel_cost

        cost[~np.isfinite(cost)] = 1e5
        row, col = linear_sum_assignment(cost)

        matches = {}
        unmatched_trks = set(trk_ids)
        unmatched_dets = set(range(len(detections)))

        for i, j in zip(row, col):
            if cost[i, j] < 0.9:
                matches[trk_ids[i]] = j
                unmatched_trks.discard(trk_ids[i])
                unmatched_dets.discard(j)

        return matches, unmatched_trks, unmatched_dets

    def _ocr_associate(self, detections, unmatched_trks, unmatched_dets):
        if not unmatched_trks or not unmatched_dets:
            return {}, unmatched_trks, unmatched_dets

        trk_ids = list(unmatched_trks)
        det_ids = list(unmatched_dets)
        cost_matrix = np.ones((len(trk_ids), len(det_ids)))

        for i, tid in enumerate(trk_ids):
            trk_box = tlwh_from_state(self.tracks[tid].mean)
            for j, did in enumerate(det_ids):
                det_box = tlwh_from_det(detections[did])
                cost_matrix[i, j] = 1 - iou(trk_box, det_box)

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
