import warnings
warnings.filterwarnings('ignore') # Masque les avertissements inoffensifs
import lightkurve as lk
import pandas as pd
import numpy as np

class AstroFetcher:
    """
    Classe responsable de la recherche et du téléchargement des courbes de lumière
    depuis les archives publiques de la NASA (Kepler/TESS).
    """

    def __init__(self, target_id):
        self.target_id = target_id

    def download_data(self, mission='Kepler'):
        """
        Cherche et télécharge les données de l'astre ciblé.
        """
        print(f"📡 Recherche des données pour l'astre {self.target_id} (Mission: {mission})...")

        # Recherche dans les bases de données de la NASA
        search_result = lk.search_lightcurve(self.target_id, mission=mission)

        if len(search_result) == 0:
            raise ValueError(f"❌ Aucune donnée trouvée pour {self.target_id}. Vérifiez l'identifiant.")

        print(f"✅ Données trouvées. Téléchargement du premier jeu d'observations...")

        # Téléchargement de la première courbe de lumière disponible
        raw_lightcurve = search_result[0].download()

        return raw_lightcurve


class SignalCleaner:
    """
    Classe responsable du prétraitement (nettoyage, retrait des valeurs aberrantes
    et gestion des trous d'observation) pour préparer les données pour l'algorithme.
    """

    def __init__(self, raw_lightcurve):
        self.lc = raw_lightcurve

    def process_data(self):
        """
        Nettoie la courbe de lumière et la convertit en DataFrame Pandas
        pour faciliter le travail de Machine Learning.
        """
        print("🧹 Nettoyage des données en cours...")

        # 1. Enlever les "trous" (Valeurs manquantes ou NaN)
        clean_lc = self.lc.remove_nans()

        # 2. Lissage et suppression des erreurs de caméra (outliers) extrêmes
        # flatten() enlève la tendance à long terme de l'étoile
        # remove_outliers(sigma=5) supprime les points impossibles physiquement
        flat_lc = clean_lc.flatten(window_length=401).remove_outliers(sigma=5)

        # 3. Conversion en DataFrame pour faciliter le travail de Jules (IA)
        df = pd.DataFrame({
            'time': flat_lc.time.value,
            'flux': flat_lc.flux.value
        })

        print("✨ Nettoyage terminé. Données prêtes pour l'analyse.")
        return df
