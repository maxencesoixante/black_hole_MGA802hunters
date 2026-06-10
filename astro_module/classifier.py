class EventClassifier:
    def __init__(self, anomaly_group):
        self.group = anomaly_group  # Un DataFrame contenant un événement spécifique

    def classify(self):
        # 1. Analyse du signe
        if self.group['flux'].mean() > 0:
            return "Éruption stellaire"

        # 2. Analyse de la forme (ex: asymétrie)
        # Transit = creux en U, Trou noir = cloche symétrique
        # Une simple mesure de skewness (asymétrie) peut aider
        skewness = self.group['flux'].skew()

        if abs(skewness) < 0.5:
            return "Trou noir (Microlentille)"
        else:
            return "Transit d'exoplanète"