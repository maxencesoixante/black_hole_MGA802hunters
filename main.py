from astro_module.data_handler import AstroFetcher, SignalCleaner
from astro_module.anomaly_engine import AnomalyDetector, AnomalyClassifier
"""from astro_module.visualizer import AstroPlotter"""

# ─────────────────────────────────────────────
# 1. Interaction utilisateur
# ─────────────────────────────────────────────
target_id = input("Entrez l'identifiant de l'astre (ex: KIC 11958492) : ")

# ─────────────────────────────────────────────
# 2. Maxence — Récupération et nettoyage
# ─────────────────────────────────────────────
fetcher = AstroFetcher(target_id)
raw_data = fetcher.download_data()
clean_data = SignalCleaner(raw_data).process_data()

# DEBUG — à supprimer après diagnostic
print("\n--- DEBUG FLUX ---")
print(f"Min flux  : {clean_data['flux'].min():.6f}")
print(f"Max flux  : {clean_data['flux'].max():.6f}")
print(f"Median    : {clean_data['flux'].median():.6f}")
print(f"Mean      : {clean_data['flux'].mean():.6f}")
print(f"Std       : {clean_data['flux'].std():.6f}")
print(f"Points < 0.999 (dips potentiels) : {(clean_data['flux'] < 0.999).sum()}")
print(f"Points > 1.001 (spikes potentiels): {(clean_data['flux'] > 1.001).sum()}")
print("------------------\n")

# ─────────────────────────────────────────────
# 3. Jules — Détection et classification
# ─────────────────────────────────────────────
detector = AnomalyDetector(clean_data, window=200, sigma_threshold=3.0)
df_flagged = detector.detect()
segments = detector.get_anomaly_segments()

classified_results = AnomalyClassifier().classify_all(segments)

# ─────────────────────────────────────────────
# 4. Alexandre — Visualisation
# ─────────────────────────────────────────────
#plotter = AstroPlotter(raw_data, clean_data, df_flagged, classified_results)
#plotter.show_results()
"""
sphinx :
You should now populate your master file path\black_hole_MGA802hunters\source\index.rst and create other documentation
source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.
"""