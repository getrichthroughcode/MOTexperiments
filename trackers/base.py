# trackers/base.py

from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseTracker(ABC):
    """
    Interface de base pour tous les trackers.
    Chaque tracker doit implémenter ces méthodes.
    """

    @abstractmethod
    def update(self, detections: List[dict]) -> None:
        """
        Met à jour les pistes avec les détections de la frame courante.
        Chaque détection est un dict contenant au minimum :
        - 'bbox' : (x, y, w, h)
        - 'score' : float
        - 'measurement' : format [x, y, a, h] pour le Kalman (optionnel selon tracker)
        """
        pass

    @abstractmethod
    def tracks_as_mot(self, frame_idx: int) -> List[Tuple[int, int, float, float, float, float, float, float, float, float]]:
        """
        Retourne les tracks au format MOTChallenge :
        [frame, id, x, y, w, h, score, -1, -1, -1]
        """
        pass
