# Rapport Final de Projet
## Détecteur Automatisé d'Anomalies Astronomiques
### Cours MGA 802 — Introduction à la Programmation en Python

---

| | |
|---|---|
| **Équipe** | Maxence Dubois · Jules Hua · Alexandre Sandre |
| **Session** | Hiver 2026 |
| **Dépôt** | `black_hole_MGA802hunters` |
| **Données** | NASA KEPLER & TESS (API `lightkurve`) |
| **Langage** | Python 3.14 — paradigme orienté objet |

---

## Table des matières

1. [Enjeux et Problématiques du Projet](#1-enjeux-et-problématiques-du-projet)
2. [Architecture Technique et Développement](#2-architecture-technique-et-développement)
   - 2.1 [Module de Maxence — Extraction et Nettoyage](#21-module-de-maxence--extraction-et-nettoyage)
   - 2.2 [Module de Jules — Détection et Classification](#22-module-de-jules--détection-et-classification)
   - 2.3 [Module d'Alexandre — Visualisation et Interface](#23-module-dalexandre--visualisation-et-interface)
3. [Observations, Apprentissages et Résultats](#3-observations-apprentissages-et-résultats)
4. [Plan de la Présentation Orale](#4-plan-de-la-présentation-orale)

---

## 1. Enjeux et Problématiques du Projet

### 1.1 Contexte astrophysique

L'astronomie moderne est confrontée à un défi fondamental : **l'excès de données**. Les missions spatiales KEPLER (2009–2018) et TESS (2018–présent) ont observé respectivement plus de 500 000 et 400 000 étoiles, produisant des dizaines de téraoctets de courbes de lumière (*light curves*) — des séries temporelles mesurant la luminosité d'une étoile sur des mois ou des années. Dans ces données se cachent des événements astrophysiques rares et précieux :

- **Les transits d'exoplanètes** : lorsqu'une planète passe devant son étoile, elle bloque une infime fraction de la lumière, provoquant une diminution périodique et symétrique du flux lumineux mesuré (typiquement −0,01 % à −1 %).
- **Les éruptions stellaires (*flares*)** : des explosions magnétiques à la surface d'une étoile libèrent une bouffée d'énergie en quelques minutes, puis se dissipent exponentiellement sur quelques heures. La signature est un pic asymétrique : montée abrupte, descente lente.
- **La microlentille gravitationnelle** : un objet massif (étoile, trou noir, naine brune) passant entre nous et l'étoile cible agit comme une lentille et amplifie temporairement sa lumière pendant plusieurs jours selon une cloche symétrique (profil Paczyński). C'est l'une des seules méthodes capables de détecter des trous noirs isolés.

### 1.2 Le problème du traitement manuel

Avant l'automatisation, la détection de ces événements reposait sur des approches de **science citoyenne** (*Citizen Science*), notamment la plateforme Zooniverse et son projet Planet Hunters : des milliers de bénévoles passaient leurs soirées à inspecter visuellement les courbes de lumière pour y repérer des anomalies. Si cette démarche a permis des découvertes réelles, elle souffre de limitations structurelles majeures :

| Limitation | Impact |
|---|---|
| Vitesse de traitement humaine | ~1 courbe / minute par personne ; impossibilité de couvrir l'ensemble des archives |
| Biais de confirmation | Les observateurs tendent à surclasser les événements ressemblant à ce qu'ils connaissent |
| Manque de reproductibilité | Deux observateurs différents peuvent classer le même événement différemment |
| Coût humain | Nécessite des dizaines de milliers d'heures-bénévoles pour quelques pourcents des données |

### 1.3 Notre réponse : un pipeline automatisé en Python

Notre projet répond directement à ce besoin en proposant une **bibliothèque Python orientée objet** qui automatise l'intégralité de la chaîne de traitement :

```
Données brutes NASA  →  Extraction  →  Nettoyage  →  Détection  →  Classification  →  Visualisation
   (API Lightkurve)      AstroFetcher   SignalCleaner  AnomalyDetector  AnomalyClassifier  AstroPlotter
```

L'objectif est de traiter en quelques secondes ce qui prendrait des heures à un observateur humain, avec une reproductibilité totale (les mêmes paramètres produisent toujours les mêmes résultats) et une sensibilité ajustable selon le type d'étoile analysé.

---

## 2. Architecture Technique et Développement

### 2.1 Diagramme de classes (UML simplifié)

```
┌─────────────────────┐        ┌──────────────────────┐
│    AstroFetcher     │        │    SignalCleaner      │
├─────────────────────┤        ├──────────────────────┤
│ - target_id : str   │        │ - data               │
├─────────────────────┤        ├──────────────────────┤
│ + download_data()   │──────▶ │ + process_data()     │
│ + load_local_csv()  │        │ + save_to_csv()      │
└─────────────────────┘        └──────────┬───────────┘
         Maxence                           │ pd.DataFrame
                                           ▼
                              ┌──────────────────────┐
                              │   AnomalyDetector    │
                              ├──────────────────────┤
                              │ - df, window, sigma  │
                              ├──────────────────────┤
                              │ + detect()           │
                              │ + get_anomaly_       │
                              │   segments()         │
                              └──────────┬───────────┘
                                         │ list[dict]
                                         ▼
                              ┌──────────────────────┐
                              │  AnomalyClassifier   │
                              ├──────────────────────┤
                              │ (constantes de seuil)│
                              ├──────────────────────┤
                              │ + classify_segment() │
                              │ + classify_all()     │
                              └──────────┬───────────┘
                                    Jules │ list[dict]
                                         ▼
                              ┌──────────────────────┐
                              │    AstroPlotter      │
                              ├──────────────────────┤
                              │ - df_flagged         │
                              │ - classified_results │
                              ├──────────────────────┤
                              │ + show_results()     │
                              └──────────────────────┘
                                      Alexandre
```

Le flux de données est **unidirectionnel** : chaque classe reçoit la sortie de la classe précédente et y ajoute de la valeur, sans jamais modifier les données des étapes antérieures. Cette architecture garantit la **séparation des responsabilités** et facilite le débogage isolé de chaque composant.

---

### 2.1 Module de Maxence — Extraction et Nettoyage

**Fichier :** `astro_module/data_handler.py`

#### La classe `AstroFetcher` — Accès aux données

`AstroFetcher` encapsule toute la logique de récupération des données, qu'elles proviennent de l'API NASA ou d'un fichier local. Son constructeur illustre deux concepts fondamentaux de la POO :

```python
class AstroFetcher:

    def __init__(self, target_id):
        # 'self' est la référence à l'objet en cours de création.
        # self.target_id = target_id "colle" la valeur sur cet objet spécifique
        # pour qu'elle soit accessible dans toutes les autres méthodes.
        self.target_id = target_id
```

**Concept POO — Le constructeur `__init__` :** c'est la méthode spéciale (dite *dunder*) que Python appelle automatiquement lors de la création d'un objet : `fetcher = AstroFetcher("KIC 11904151")`. Son rôle est d'initialiser l'état interne de l'objet (ses *attributs*).

**Concept POO — `self` :** `self` est la référence à *cette* instance spécifique de la classe. Si l'on crée deux `AstroFetcher` différents (pour deux étoiles différentes), chacun possède son propre `self.target_id` indépendant.

La méthode `download_data` implémente une logique de **repli automatique** (*fallback*) entre missions :

```python
def download_data(self, mission='Kepler'):
    missions_to_try = [mission, 'TESS'] if mission.lower() == 'kepler' \
                      else [mission, 'Kepler']

    for current_mission in missions_to_try:
        try:
            search_result = lk.search_lightcurve(
                self.target_id, mission=current_mission
            )
        except Exception as e:
            print(f"  Warning: {e}")
            search_result = []

        if len(search_result) > 0:
            return search_result[0].download()

    raise ValueError(f"No data found for '{self.target_id}'.")
```

Ce pattern `try / except` est essentiel pour un programme robuste : une panne réseau ou un identifiant mal résolu ne fait pas planter l'application, elle tente une alternative et communique clairement l'échec si toutes les options sont épuisées.

#### La classe `SignalCleaner` — Prétraitement mathématique

`SignalCleaner` contient le travail mathématique le plus avancé du projet. Sa méthode `process_data` est un exemple de **polymorphisme** : elle adapte automatiquement son comportement selon le type de données reçues (objet Lightkurve ou DataFrame Pandas), sans que l'appelant ait à s'en préoccuper.

```python
def process_data(self):
    # isinstance() détecte le type de l'entrée et branche vers la bonne logique.
    if isinstance(self.data, pd.DataFrame):
        # ── Chemin A : CSV local ────────────────────────────────────────────
        df = self.data.dropna(subset=['time', 'flux'])

        # Règle des 5 sigma : suppression des valeurs aberrantes extrêmes
        mean_flux = df['flux'].mean()
        std_flux  = df['flux'].std()
        df = df[
            (df['flux'] >= mean_flux - 5 * std_flux) &
            (df['flux'] <= mean_flux + 5 * std_flux)
        ].copy()

        # Normalisation par la médiane
        median_flux = df['flux'].median()
        df['flux']  = df['flux'] / (median_flux + 1e-10)
        return df.reset_index(drop=True)

    else:
        # ── Chemin B : objet Lightkurve (API NASA) ──────────────────────────
        clean_lc    = self.data.remove_nans().remove_outliers(sigma=5)
        flux_values = clean_lc.flux.value
        median_flux = float(np.median(flux_values))
        df = pd.DataFrame({
            'time': clean_lc.time.value,
            'flux': flux_values / (median_flux + 1e-10)
        })
        return df.reset_index(drop=True)
```

**Justification mathématique de la règle des 5σ :**

Soit $\mu$ la moyenne et $\sigma$ l'écart-type du flux sur l'ensemble de la courbe de lumière. Pour une distribution gaussienne (modèle raisonnable du bruit instrumental), la probabilité qu'un point réel dépasse $\pm 5\sigma$ est inférieure à $6 \times 10^{-7}$, soit environ 1 chance sur 1,7 million. Tout point au-delà de cette limite est donc, avec une certitude quasi-absolue, un artefact (rayon cosmique, saturation du CCD, corruption de données) et non un signal astrophysique.

$$P(|X - \mu| > 5\sigma) < 6 \times 10^{-7}$$

**Justification du choix de la médiane pour la normalisation :**

Nous divisons chaque mesure de flux par la médiane plutôt que par la moyenne. La médiane est **robuste aux valeurs extrêmes** : si une éruption stellaire fait monter le flux à 150 % pendant 1 heure sur 90 jours d'observation, la médiane reste proche de la vraie valeur de repos de l'étoile, tandis que la moyenne serait légèrement biaisée vers le haut. Après normalisation, toute la courbe est ramenée autour de 1,0, ce qui rend les comparaisons entre différentes étoiles (de luminosités très différentes) directement possibles.

$$\text{flux\_normalisé} = \frac{\text{flux\_brut}}{\text{médiane(flux)} + \varepsilon}$$

où $\varepsilon = 10^{-10}$ est un terme de garde contre la division par zéro.

**Décision de ne pas utiliser `flatten()` :**

La bibliothèque Lightkurve propose une méthode `flatten()` qui ajuste et soustrait une tendance lisse sur une large fenêtre (401 points). Si cette méthode élimine efficacement les dérives instrumentales lentes, elle supprime également les creux de transit d'exoplanètes : un dip de ~15 points ressemble à une tendance courte aux yeux du lisseur. Notre approche par normalisation médiane préserve intégralement la profondeur relative des transits, qui représente typiquement 0,01 % à 1 % du flux.

---

### 2.2 Module de Jules — Détection et Classification

**Fichier :** `astro_module/anomaly_engine.py`

#### La classe `AnomalyDetector` — Détection par fenêtre glissante

L'algorithme central du projet est une **détection par fenêtre glissante** (*rolling window*). Plutôt que de comparer chaque point à une moyenne globale de la série entière (ce qui serait insensible aux dérives instrumentales lentes), l'algorithme calcule à chaque position une moyenne et un écart-type **locaux** sur les 200 points voisins.

```python
def detect(self) -> pd.DataFrame:
    # Moyenne mobile centrée (baseline locale)
    self.df['rolling_mean'] = (
        self.df['flux']
        .rolling(window=self.window, center=True, min_periods=1)
        .mean()
    )
    # Écart-type mobile (variabilité locale)
    self.df['rolling_std'] = (
        self.df['flux']
        .rolling(window=self.window, center=True, min_periods=1)
        .std()
    )
    # Enveloppe de détection dynamique
    upper = self.df['rolling_mean'] + self.sigma_threshold * self.df['rolling_std']
    lower = self.df['rolling_mean'] - self.sigma_threshold * self.df['rolling_std']

    # Drapeau booléen : True = point hors enveloppe = anomalie
    self.df['is_anomaly'] = (
        (self.df['flux'] > upper) | (self.df['flux'] < lower)
    )
    return self.df
```

**Justification mathématique du seuil de détection :**

Pour chaque point d'indice $i$, on calcule :

$$\mu_i = \frac{1}{W} \sum_{j=i-W/2}^{i+W/2} \text{flux}_j \qquad \sigma_i = \sqrt{\frac{1}{W-1} \sum_{j=i-W/2}^{i+W/2} (\text{flux}_j - \mu_i)^2}$$

Un point est marqué comme anomalie si :

$$|\text{flux}_i - \mu_i| > N \cdot \sigma_i$$

où $N$ est le paramètre `sigma_threshold`. L'avantage de cette formulation **dynamique** est que le seuil s'adapte automatiquement à la variabilité locale : dans une région calme (petit $\sigma_i$), même une légère déviation est flaggée ; dans une région naturellement variable (grand $\sigma_i$), le seuil est élargi pour éviter les faux positifs.

**Paramètre `window = 200` :** avec des données Kepler à cadence longue (30 min/point), 200 points représentent ~4 jours d'observation, ce qui donne une baseline lisse capable de suivre les dérives instrumentales sur l'échelle de la semaine, sans être biaisée par un transit individuel (~6-10 heures).

#### La méthode `get_anomaly_segments` — Regroupement temporel

Les points anormaux isolés ne constituent pas des événements physiques — un transit, par exemple, laisse une empreinte sur ~10-30 points consécutifs. La méthode `get_anomaly_segments` regroupe les points anormaux en **segments cohérents** :

```python
def get_anomaly_segments(self, min_gap_days: float = 0.5):
    anomaly_indices = self.df.index[self.df['is_anomaly']].tolist()
    times           = self.df.loc[anomaly_indices, 'time'].values

    segments        = []
    current_segment = [anomaly_indices[0]]

    for i in range(1, len(anomaly_indices)):
        time_gap = times[i] - times[i - 1]   # gap en jours

        if time_gap <= min_gap_days:
            current_segment.append(anomaly_indices[i])  # même événement
        else:
            segments.append(current_segment)             # nouvel événement
            current_segment = [anomaly_indices[i]]

    segments.append(current_segment)
    # ... ajout du contexte (100 points de chaque côté)
```

Le regroupement se fait par **proximité temporelle** (gap < 0,5 jour = 12h) et non par proximité d'indice, car les trous d'observation du télescope créent des discontinuités dans les indices sans interruption physique réelle de l'événement.

#### La classe `AnomalyClassifier` — Arbre de décision multi-critères

Chaque segment est ensuite soumis à un arbre de décision basé sur quatre caractéristiques (*features*) extraites de la forme du signal :

```python
class AnomalyClassifier:
    DIP_THRESHOLD              = -0.0008   # dip en dessous → TRANSIT
    ASYMMETRY_THRESHOLD        = 0.30      # montée rapide → FLARE
    MIN_MICROLENSING_AMPLITUDE = 0.005     # amplitude min → MICROLENSING
    MIN_MICROLENSING_DURATION  = 0.3       # durée min (j) → MICROLENSING
```

**Feature 1 — Direction :** le pic est-il un creux (flux sous la baseline) ou un pic de brillance ?

```python
baseline  = np.median(flux)
norm_flux = (flux - baseline) / (np.abs(baseline) + 1e-10)
peak_idx  = int(np.argmax(np.abs(norm_flux)))
is_dip    = norm_flux[peak_idx] < self.DIP_THRESHOLD
```

**Feature 2 — Asymétrie de pente :** la montée est-elle plus rapide que la descente ?

```python
left_flux       = norm_flux[:peak_idx + 1]
right_flux      = norm_flux[peak_idx:]
rise_slope      = np.mean(np.abs(np.diff(left_flux)))
decay_slope     = np.mean(np.abs(np.diff(right_flux)))
slope_asymmetry = (rise_slope - decay_slope) / (rise_slope + decay_slope + 1e-10)
```

Cette métrique varie entre −1 (descente instantanée, montée lente) et +1 (montée instantanée, descente lente), ce qui est caractéristique des éruptions stellaires.

**Feature 3 et 4 — Amplitude et durée :** un signal trop faible ou trop court est du bruit résiduel, pas un événement réel.

**Arbre de décision (première règle vérifiée gagne) :**

```
is_dip ?
  └── OUI → TRANSIT   (probabilité d'exoplanète, symétrie en bonus de confiance)
  └── NON → slope_asymmetry > 0.30 ?
              └── OUI → FLARE    (montée abrupte caractéristique)
              └── NON → amplitude < 0.005 OU durée < 0.3 j ?
                          └── OUI → NOISE       (signal trop faible ou trop court)
                          └── NON → MICROLENSING (cloche symétrique ample et longue)
```

Le score de confiance est calculé différemment pour chaque type d'événement, en valorisant la caractéristique la plus discriminante : la symétrie pour les transits et la microlentille, l'asymétrie pour les flares.

---

### 2.3 Module d'Alexandre — Visualisation et Interface

**Fichiers :** `astro_module/visualizer.py` · `main.py`

#### La classe `AstroPlotter` — Communication des résultats

La visualisation est le seul point de contact entre notre code et un utilisateur humain. Une mauvaise visualisation peut rendre des résultats corrects incompréhensibles. `AstroPlotter` construit un graphique professionnel couche par couche :

```python
class AstroPlotter:
    COLOUR_NORMAL   = '#B0C8E8'   # bleu acier discret — cadences normales
    COLOUR_ANOMALY  = '#E84040'   # rouge vif          — cadences anormales
    COLOUR_LABELS   = {
        'TRANSIT'     : '#9B59B6',   # violet
        'FLARE'       : '#E67E22',   # orange
        'MICROLENSING': '#27AE60',   # vert
    }

    def show_results(self):
        # Couche 1 : points normaux (petits, transparents, en arrière-plan)
        ax.scatter(df_normal['time'], df_normal['flux'],
                   color=self.COLOUR_NORMAL, s=2, alpha=0.5, zorder=1)

        # Couche 2 : anomalies (grandes, opaques, au premier plan)
        ax.scatter(df_anomaly['time'], df_anomaly['flux'],
                   color=self.COLOUR_ANOMALY, s=18, alpha=0.95, zorder=3)

        # Couche 3 : baseline (ligne pointillée bleue)
        ax.plot(df['time'], df['rolling_mean'],
                color=self.COLOUR_BASELINE, linestyle='--', zorder=2)

        # Couche 4 : bande de confiance ±σ (zone ombrée)
        ax.fill_between(df['time'], lower_band, upper_band,
                        color=self.COLOUR_BAND, alpha=0.15, zorder=0)

        # Couche 5 : annotations fléchées pour chaque événement classifié
        for event in self.classified_results:
            ax.annotate(f"{event['event_type']}\n({event['confidence']:.0%})",
                        xy=(peak_time, peak_flux),
                        xytext=(peak_time, peak_flux + 0.03),
                        arrowprops=dict(arrowstyle='->', color=colour))
```

Les **attributs de classe** (les constantes `COLOUR_*`) illustrent une subtilité POO importante : ils sont partagés par toutes les instances de `AstroPlotter`. Changer `COLOUR_ANOMALY = '#FF6600'` suffit à modifier la couleur dans tout le projet sans toucher à la logique.

#### Le script `main.py` — Orchestration et robustesse

`main.py` assemble les trois modules et gère l'interaction utilisateur via une **boucle de reprise** (*retry loop*) qui évite les crashes :

```python
while True:
    target_id = input("Enter the star identifier: ").strip()

    if target_id.lower() in ('quit', 'exit', 'q'):
        raise SystemExit(0)

    try:
        fetcher    = AstroFetcher(target_id)
        raw_data   = fetcher.download_data(mission='Kepler')
        clean_data = SignalCleaner(raw_data).process_data()
        break   # succès → sortir de la boucle

    except ValueError as e:
        print(f"\n[Error] {e}")
        print("Please try a different identifier.\n")
        # la boucle recommence automatiquement
```

Le pattern `try / except / break` est fondamental dans les programmes interactifs robustes : il transforme une erreur fatale (crash du programme) en un état récupérable (message d'erreur clair, nouvelle tentative).

---

## 3. Observations, Apprentissages et Résultats

### 3.1 Ce que nous avons réussi à produire

Notre équipe a livré un **pipeline de données fonctionnel de bout en bout** :

- ✅ Connexion et téléchargement automatisé depuis l'API NASA via `lightkurve`
- ✅ Chargement alternatif depuis un fichier CSV local (données Kaggle)
- ✅ Nettoyage adaptatif (deux formats d'entrée gérés de manière transparente)
- ✅ Détection par fenêtre glissante avec seuil dynamique ajustable
- ✅ Classification automatique en 4 catégories (TRANSIT, FLARE, MICROLENSING, NOISE)
- ✅ Visualisation annotée professionnelle avec code couleur par type d'événement
- ✅ Interface CLI interactive avec gestion d'erreurs et boucle de reprise
- ✅ Script de validation sur Kepler-22 (`test_kepler22.py`)

### 3.2 Défis techniques rencontrés

#### Défi 1 — Les trous d'observation du télescope

Les satellites KEPLER et TESS interrompent leurs observations régulièrement (manœuvres de pointage, mise en veille sécurisée, téléchargement des données vers la Terre). Ces interruptions créent des **trous temporels** dans la série chronologique : l'indice de ligne avance, mais le temps physique fait un saut de plusieurs heures ou jours.

Notre première implémentation de `get_anomaly_segments` regroupait les points anormaux par **proximité d'indice** (si deux points anormaux sont à moins de 10 indices l'un de l'autre, ils appartiennent au même événement). Cette approche échouait systématiquement sur les données réelles : un transit de 8 heures de part et d'autre d'un trou d'observation était séparé en deux faux événements distincts.

**Solution implémentée :** regroupement par **proximité temporelle** (gap < 0,5 jour = 12 heures), indépendamment des indices. Cette modification, en apparence mineure, a éliminé la grande majorité des faux segments.

#### Défi 2 — Le choix de l'hyperparamètre sigma

La valeur du seuil de détection `sigma_threshold` est critique et non universelle :

| Type d'étoile | sigma recommandé | Risque à 3,0 |
|---|---|---|
| Étoile solaire calme (ex: Kepler-22) | 3,0 – 4,0 | Acceptable |
| Naine M active (ex: GJ 1243) | 5,0 ou plus | ~300 faux positifs MICROLENSING |

Notre test sur **GJ 1243** a mis en évidence ce problème : cette étoile M-naine à rotation rapide (~1,6 jour) présente une variabilité naturelle qui dépasse en permanence le seuil à 3σ. Le détecteur a identifié 299 événements MICROLENSING correspondant en réalité aux bosses de la rotation stellaire. Le passage à sigma=5,0 sur **Kepler-22** restaure un résultat propre avec seulement quelques transits détectés.

**Ce que cela nous a appris :** un algorithme universel n'existe pas en astrophysique. La qualité du résultat dépend toujours du contexte physique de la cible. Documenter les limites de son algorithme est aussi important que de présenter ses succès.

#### Défi 3 — Polymorphisme et séparation des responsabilités

La bibliothèque `lightkurve` retourne un objet de type `LightCurve` (avec des unités AstroPy) lorsqu'on télécharge depuis l'API, mais nos propres fichiers CSV retournent un DataFrame Pandas. Ces deux formats sont incompatibles au niveau de l'API. Nous devions faire un choix d'architecture :

- **Option A** : imposer à l'utilisateur de convertir lui-même avant d'appeler `SignalCleaner` → mauvaise expérience utilisateur, code verbeux dans `main.py`.
- **Option B** : créer deux classes séparées, `LightcurveCleaner` et `CSVCleaner` → duplication de code importante.
- **Option C (choisie)** : un seul `SignalCleaner` qui détecte le type avec `isinstance()` et branche vers la logique appropriée → interface unifiée, code réutilisable.

Cette décision illustre le principe de **responsabilité unique** inversé : le module qui connaît les deux formats est la meilleure entité pour gérer la traduction, pas l'appelant.

### 3.3 Ce que le projet nous a appris

**Sur la programmation orientée objet :**

- Un `__init__` n'est pas là pour « faire quelque chose » mais pour « préparer l'objet ». La vraie logique va dans des méthodes nommées.
- Les attributs de classe (définis en dehors de `__init__`) sont des constantes partagées entre toutes les instances — parfaits pour les seuils de l'algorithme.
- Le chaînage de méthodes (`self.data.remove_nans().remove_outliers()`) rend le code lisible comme une phrase en anglais.

**Sur la manipulation de données avec Pandas et NumPy :**

- Les opérations vectorisées (`df['flux'] / median`) sont des dizaines de fois plus rapides que des boucles `for` équivalentes sur 50 000 points.
- Les masques booléens (`df[df['flux'] > threshold]`) remplacent élégamment des blocs `if / else` complexes.
- `rolling().mean()` et `rolling().std()` en une ligne font ce qui nécessiterait 15 lignes de code manuel.

**Sur la robustesse logicielle :**

- Toujours valider les données à l'entrée (vérifier les colonnes, vérifier que `detect()` a été appelé avant `get_anomaly_segments()`).
- Les exceptions levées avec des messages clairs valent mieux qu'une exécution silencieuse sur des données incorrectes.
- Un script interactif sans `try / except` autour des entrées utilisateur crashera inévitablement en démonstration.

---

## 4. Plan de la Présentation Orale

**Durée totale estimée :** 20–25 minutes + 5 minutes de questions  
**Grille d'évaluation couverte :** Objectifs du projet · Choix de programmation · Démonstration · Tâches restantes / limites

---

### Slide 1 — Page de titre
**Durée :** 30 secondes  
**Présentateur :** les trois ensemble  

**Contenu visuel :**
- Titre : « Détecteur Automatisé d'Anomalies Astronomiques »
- Image de fond : courbe de lumière de Kepler avec un transit visible
- Noms des trois membres avec leurs rôles respectifs

**Points à l'oral :**
> *« Notre projet répond à une question concrète : comment analyser automatiquement des millions de courbes de lumière stellaires pour trouver des exoplanètes, des éruptions, ou des signatures de trous noirs — sans intervention humaine ? »*

---

### Slide 2 — Le problème : la masse de données spatiales
**Durée :** 2 minutes  
**Présentateur :** Alexandre (introduction générale, rôle de chef d'orchestre)

**Contenu visuel :**
- Infographie : 500 000 étoiles observées par KEPLER, échelle du volume de données (téraoctets)
- Photo de l'interface Zooniverse / Planet Hunters
- Tableau comparatif : traitement humain vs. traitement automatisé

**Points à l'oral :**
- Expliquer le concept de courbe de lumière (*light curve*) en termes simples : graphique de la luminosité d'une étoile dans le temps
- Expliquer pourquoi le traitement manuel est insuffisant à l'échelle des archives spatiales
- Présenter les trois types d'événements que notre algorithme cherche à détecter (transit, flare, microlentille) avec une illustration de la signature de chacun

**Critère d'évaluation visé :** *Objectifs du projet*

---

### Slide 3 — Architecture du pipeline
**Durée :** 2 minutes  
**Présentateur :** Alexandre

**Contenu visuel :**
- Diagramme de flux du pipeline : `API NASA → AstroFetcher → SignalCleaner → AnomalyDetector → AnomalyClassifier → AstroPlotter`
- Couleurs distinctes pour chaque module avec initiale du responsable
- Structure des dossiers du projet (`astro_module/`, `main.py`, `test_kepler22.py`)

**Points à l'oral :**
- Présenter la séparation des responsabilités entre les trois membres
- Expliquer pourquoi on utilise des classes plutôt que des fonctions isolées : l'objet garde la mémoire de son état entre les appels
- Mentionner que `main.py` est le « chef d'orchestre » qui assemble les modules

**Critère d'évaluation visé :** *Choix de programmation — Architecture POO*

---

### Slide 4 — Maxence : Extraction des données (AstroFetcher)
**Durée :** 2–3 minutes  
**Présentateur :** Maxence

**Contenu visuel :**
- Extrait de code : la méthode `download_data` avec la boucle de fallback Kepler → TESS
- Screenshot du terminal montrant la recherche et le téléchargement d'une courbe de lumière

**Points à l'oral :**
- Expliquer comment la bibliothèque `lightkurve` interroge l'archive NASA MAST
- Présenter le mécanisme de repli automatique (si Kepler échoue, on essaie TESS)
- Expliquer le pattern `try / except` : gérer les erreurs réseau ou les identifiants invalides sans faire crasher le programme
- Mentionner le support des fichiers CSV locaux comme alternative hors-ligne (polymorphisme de `load_local_csv`)

**Question anticipée du jury :** *« Pourquoi deux méthodes de chargement ? »* → Flexibilité : API pour les tests en temps réel, CSV pour travailler sans connexion (données Kaggle).

**Critère d'évaluation visé :** *Choix de programmation — Gestion des erreurs, API*

---

### Slide 5 — Maxence : Nettoyage du signal (SignalCleaner)
**Durée :** 3 minutes  
**Présentateur :** Maxence

**Contenu visuel :**
- Graphique avant/après nettoyage sur une même courbe (superposition ou côte-à-côte)
- Formule de la règle des 5σ et de la normalisation par la médiane
- Extrait de code : le bloc `isinstance` + le filtre sigma + la normalisation

**Points à l'oral :**
- Expliquer intuitivement la règle des 5σ : « moins d'une chance sur un million qu'un point réel soit rejeté »
- Expliquer pourquoi on normalise par la **médiane** et non la moyenne : robustesse face aux valeurs extrêmes
- Justifier le refus d'utiliser `flatten()` : cette méthode efface les creux de transit qu'on cherche justement à détecter
- Insister sur le polymorphisme : `process_data()` gère transparemment deux formats d'entrée différents

**Critère d'évaluation visé :** *Choix de programmation — Algorithmes, mathématiques*

---

### Slide 6 — Jules : Détection par fenêtre glissante (AnomalyDetector)
**Durée :** 3 minutes  
**Présentateur :** Jules

**Contenu visuel :**
- Animation ou schéma de la fenêtre glissante (rectangle qui se déplace sur la courbe)
- Formule : $\mu_i$, $\sigma_i$, condition d'anomalie $|\text{flux}_i - \mu_i| > N \cdot \sigma_i$
- Graphique montrant la courbe de lumière avec la baseline (rolling mean) et la bande ±3σ, et les points anormaux en rouge

**Points à l'oral :**
- Comparer la moyenne locale vs. globale : « imaginez chercher un pic anormal dans un concert de rock en comparant chaque note à la moyenne générale du bruit — vous rateriez tous les pics »
- Expliquer le paramètre `center=True` : la fenêtre regarde également à gauche et à droite du point courant
- Expliquer le choix de `window=200` : 200 × 30 min ≈ 4 jours, assez long pour une baseline stable, assez court pour ne pas englober plusieurs transits

**Critère d'évaluation visé :** *Choix de programmation — Algorithme de détection*

---

### Slide 7 — Jules : Classification automatique (AnomalyClassifier)
**Durée :** 3 minutes  
**Présentateur :** Jules

**Contenu visuel :**
- Diagramme de l'arbre de décision (4 branches : dip → TRANSIT, asymétrie → FLARE, trop faible/court → NOISE, sinon → MICROLENSING)
- Exemple de 3 segments côte-à-côte avec leur profil caractéristique et leur classification

**Points à l'oral :**
- Expliquer chaque feature : direction du pic, asymétrie de pente (rise vs. decay), amplitude, durée
- Expliquer le calcul de l'asymétrie : `(rise_slope - decay_slope) / (rise_slope + decay_slope)`
- Justifier les constantes de classe : pourquoi les seuils sont en attributs de classe (modifiables facilement, partagés par toutes les instances)
- Expliquer le score de confiance : il valorise la caractéristique la plus discriminante pour chaque type

**Question anticipée du jury :** *« C'est du machine learning ? »* → Non, c'est un **classifieur à règles** (*rule-based*). C'est un choix délibéré : interprétable, explicable, pas de besoin de données d'entraînement étiquetées.

**Critère d'évaluation visé :** *Choix de programmation — Algorithme de classification*

---

### Slide 8 — Alexandre : Visualisation (AstroPlotter)
**Durée :** 2 minutes  
**Présentateur :** Alexandre

**Contenu visuel :**
- Capture d'écran du graphique final généré par `AstroPlotter.show_results()` sur Kepler-10
- Légende annotée (couleurs, bande ±σ, annotations fléchées)

**Points à l'oral :**
- Expliquer les choix de design : rouge vif pour les anomalies (contraste maximal), annotations fléchées directement sur le pic (pas dans une table séparée), couleur différente par type d'événement
- Mentionner le système de couches (`zorder`) : la bande de fond, puis les points normaux, puis la baseline, puis les anomalies, puis les annotations — dans cet ordre précis
- Présenter les attributs de classe `COLOUR_*` : configuration centralisée, facile à modifier sans toucher à la logique

**Critère d'évaluation visé :** *Choix de programmation — Interface utilisateur, visualisation*

---

### Slide 9 — Démonstration live
**Durée :** 3–4 minutes  
**Présentateur :** Maxence (frappe) + Alexandre (commentaire à l'oral)

**Contenu visuel :**
- Terminal en direct : `python main.py`
- Saisie de `KIC 11904151` (Kepler-10, cible propre et connue)
- Le graphique s'affiche en direct

**Points à l'oral (Alexandre) :**
- Narrer chaque étape au fur et à mesure que les messages apparaissent dans le terminal
- Pointer les anomalies sur le graphique et lier visuellement à la classification dans le terminal
- Montrer la robustesse : entrer un identifiant invalide volontairement, montrer que le programme propose une nouvelle tentative au lieu de crasher

**Critère d'évaluation visé :** *Démonstration*

---

### Slide 10 — Résultats : cas validé (Kepler-22, sigma=5.0)
**Durée :** 2 minutes  
**Présentateur :** Jules

**Contenu visuel :**
- Capture du graphique Kepler-22 avec les transits annotés
- Tableau de résultats du script `test_kepler22.py` : nombre d'anomalies, répartition par type

**Points à l'oral :**
- Présenter les résultats quantitatifs (nombre d'événements classifiés, confiance moyenne)
- Expliquer pourquoi sigma=5.0 est plus adapté à cette étoile que sigma=3.0
- Lier visuellement les transits détectés à la période connue de Kepler-22b (~290 jours)

**Critère d'évaluation visé :** *Démonstration — Résultats*

---

### Slide 11 — Limites de l'approche : le cas GJ 1243
**Durée :** 2 minutes  
**Présentateur :** Jules

**Contenu visuel :**
- Graphique GJ 1243 (sigma=3.0) avec les ~300 faux positifs MICROLENSING
- Graphique de la rotation stellaire détectée (période 0,83 j ≈ moitié de la vraie période)
- Tableau comparatif : GJ 1243 (sigma=3.0) vs. Kepler-22 (sigma=5.0)

**Points à l'oral :**
- Expliquer honnêtement le problème : la variabilité naturelle de GJ 1243 dépasse en permanence le seuil à 3σ
- Expliquer le phénomène d'alias de période : on détecte 0,83 j (moitié de la vraie période de 1,6 j) car une bosse sur deux correspond à la géométrie de l'angle de vue
- Conclure que ce n'est pas un bug mais une **limite connue de l'approche** : le rolling window + sigma est calibré pour les étoiles calmes

> *« Un algorithme honnête doit savoir reconnaître ses limites. GJ 1243 nous montre que le même outil ne fonctionne pas sur toutes les étoiles avec les mêmes paramètres. »*

**Critère d'évaluation visé :** *Tâches restantes — Limites identifiées*

---

### Slide 12 — Conclusion et perspectives
**Durée :** 1–2 minutes  
**Présentateur :** Maxence (bilan) + proposition de suite par les trois

**Contenu visuel :**
- Récapitulatif du pipeline en une ligne par étape (✅ fait / 🔲 à faire)
- Trois axes d'amélioration future

**Points à l'oral :**

Bilan de l'existant :
- Pipeline complet, fonctionnel, testé sur des données réelles NASA
- Architecture modulaire et maintenable (chaque classe est remplaçable)
- Code entièrement documenté et commenté

Pistes d'amélioration :
1. **Améliorations algorithmiques** : intégration d'un modèle de Machine Learning (Random Forest sur les features extraites) pour remplacer l'arbre de règles et s'adapter à un plus grand nombre de types d'étoiles
2. **Données multi-quartiers** : stitcher toutes les observations disponibles pour une même étoile afin d'augmenter la probabilité de capturer plusieurs transits et de valider la périodicité
3. **Interface utilisateur** : remplacer le terminal par une interface web légère (Streamlit) qui permettrait à n'importe quel utilisateur de lancer une analyse sans connaître Python

**Critère d'évaluation visé :** *Tâches restantes — Ouvertures*

---

## Annexes

### A. Paramètres de configuration recommandés par type d'étoile

| Type d'étoile | Exemple | sigma recommandé | window recommandé |
|---|---|---|---|
| Solaire calme | Kepler-10, Kepler-22 | 3.0 | 200 |
| Géante / sous-géante | KIC 5110407 | 4.0 | 300 |
| Naine M active | GJ 1243, GJ 876 | 5.0 ou plus | 150 |
| Étoile à pulsations | Delta Cephei | Non recommandé | — |

### B. Dépendances du projet

```
lightkurve >= 2.4
pandas     >= 2.0
numpy      >= 1.26
matplotlib >= 3.8
seaborn    >= 0.13  (optionnel — styling uniquement)
```

### C. Commandes de lancement

```bash
# Pipeline interactif complet
python main.py

# Test de validation Kepler-22 (sigma=5.0)
python test_kepler22.py

# Test d'intégration minimal (Maxence — Kepler-10)
python data/raw/test_data.py
```
