from astro_modules.data_handler import AstroFetcher, SignalCleaner

# 1. On cible l'étoile Kepler-10 (identifiant KIC 11904151)
fetcher = AstroFetcher("KIC 11904151")

# 2. On télécharge la donnée
donnees_brutes = fetcher.download_data(mission='Kepler')

# 3. On nettoie la donnée
cleaner = SignalCleaner(donnees_brutes)
donnees_propres = cleaner.process_data()

# 4. On vérifie que Pandas a bien fait son travail
print("\nVoici les 5 premières lignes des données prêtes pour l'IA :")
print(donnees_propres.head())