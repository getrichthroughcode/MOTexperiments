# **MOT Experiments** 

Ce projet implémente plusieurs trackers de type tracking-by-detection évalués sur le benchmark MOT17.
Il fournit un script d’évaluation unique permettant de comparer différentes approches comme OC‑SORT, ByteTrack ou un tracker personnalisé basé sur un filtre de Kalman.

## **Description**

Le dépôt contient :
	-	un script d’évaluation compatible avec les fichiers MOTChallenge,
	-	une interface commune pour implémenter facilement de nouveaux trackers,
	-	plusieurs implémentations de référence (OC‑SORT, ByteTrack),
	-	des utilitaires pour charger et préparer les données.

L’objectif est d’expérimenter rapidement de nouvelles stratégies de suivi et de mesurer leurs performances à l’aide des métriques MOTChallenge (MOTA, IDF1, IDP, IDR, etc.).

### **Structure du repo**
```plaintext
├── data
│   └── results
│       └── summary.txt
├── LICENSE
├── Makefile
├── models
│   ├── __init__.py
│   └── kalman.py
├── mot17_tracking_experiment.py
├── README.md
├── requirements.txt
├── trackers
│   ├── __init__.py
│   ├── base.py
│   ├── bytetrack.py
│   ├── customtrackers.py
│   └── ocsort.py
└── utils
    ├── __init__.py
    └── io.py
```
```
```

## **Installation**
Clonez le projet et installez les dépendances :
```bash
git clone https://github.com/getrichthroughcode/MOTexperiments/tree/main
cd MOTexperiments
pip install -r requirements.txt
```

## **Ajouter un Tracker**

	1.	Créez un fichier dans trackers/ (par exemple montracker.py).
	2.	Implémentez une classe qui hérite de BaseTracker et redéfinissez au minimum :
      -  update(self, detections: List[dict])
      -	tracks_as_mot(self, frame_idx: int)
	3.	Importez et utilisez votre tracker dans mot17_tracking_experiment.py.





```
```
```
