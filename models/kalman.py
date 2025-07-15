import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """
    Un filtre de Kalman pour le suivi de bounding boxes.
    Le vecteur d'état est de dimension 8 :
        [x, y, a, h, vx, vy, va, vh]
    où (x, y) est le centre, a le ratio largeur/hauteur et h la hauteur.
    """

    def __init__(self):
        ndim, dt = 4, 1.4
        self.F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = 1
        self.H = np.eye(ndim, 2 * ndim)
        self.sigma_p = 1.0 / 20  # incertitude sur la position
        self.sigma_v = 1.0 / 160  # incertitude sur la vitesse

    def initialiser(self, mesures):
        """
        Création d'un track à partir d'une mesure (format [x, y, a, h]).
        """
        mean_pos = mesures
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def initialiser(self, mesures):
        """
        Création d'un track à partir d'une mesure (format [x, y, a, h]).
        """
        mean_pos = mesures
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            2 * self.sigma_p * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
            10 * self.sigma_v * mesures[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def prediction(self, mean, covariance):
        """
        Étape de prédiction.
        """
        std_pos = [2 * self.sigma_p * mean[3]] * 4
        std_vel = [2 * self.sigma_v * mean[3]] * 4
        cov_noise = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self.F, mean)
        covariance = np.linalg.multi_dot((self.F, covariance, self.F.T)) + cov_noise
        return mean, covariance

    def project(self, mean, covariance):
        """
        Passage de l'espace état à l'espace de mesure.
        """
        std = [self.sigma_p * mean[3]] * 4
        innovation_cov = np.diag(np.square(std))
        mean_proj = np.dot(self.H, mean)
        covariance_proj = np.linalg.multi_dot((self.H, covariance, self.H.T))
        return mean_proj, covariance_proj + innovation_cov

    def update(self, mean, covariance, mesures):
        """
        Mise à jour de l'état avec une nouvelle mesure.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.H.T).T, check_finite=False
        ).T
        innovation = mesures - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self, mean, covariance, mesures, only_position=False, metric="maha"
    ):
        """
        Calcule la distance (au carré) de Mahalanobis entre la prédiction et des mesures.
        Les mesures sont attendues au format (Nx4), ici [x, y, a, h].
        """
        mean_proj, covariance_proj = self.project(mean, covariance)
        if only_position:
            mean_proj, covariance_proj = mean_proj[:2], covariance_proj[:2, :2]
            mesures = mesures[:, :2]
        d = mesures - mean_proj
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance_proj)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("Invalid distance metric")
