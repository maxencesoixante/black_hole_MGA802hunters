import pandas as pd
import numpy as np


class AnomalyDetector:
    def __init__(self, dataframe):
        self.df = dataframe
        # Calcul de la ligne de base (moyenne mobile)
        # On utilise une fenêtre pour lisser et ignorer le bruit local
        self.df['baseline'] = self.df['flux'].rolling(window=50, center=True).median()

    def identify_anomalies(self, sigma_threshold=3):
        """
        Détecte les points où le flux s'éloigne trop de la baseline.
        """
        # Calcul de l'écart type local pour définir la sensibilité
        std = self.df['flux'].std()

        # Calcul du Z-score : (valeur - moyenne) / std
        self.df['z_score'] = (self.df['flux'] - self.df['baseline']) / std

        # Détection : Si le Z-score dépasse le seuil, c'est une anomalie
        self.df['is_anomaly'] = np.abs(self.df['z_score']) > sigma_threshold

        return self.df[self.df['is_anomaly']]

    def get_anomaly_segments(self):
        """
        Regroupe les points d'anomalie successifs pour former des 'événements'.
        """
        # Logique pour regrouper les indices consécutifs en segments d'anomalies
        # (À implémenter pour faciliter la classification par type)
        pass