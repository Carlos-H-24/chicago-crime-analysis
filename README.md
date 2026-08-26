# Chicago Crime & Weather Analysis

> Analyse de l'influence des conditions météorologiques et socio-économiques sur la
> criminalité à Chicago — 500 000 observations sur 20 ans — avec un modèle prédictif
> d'arrestation et un dashboard interactif.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Data](https://img.shields.io/badge/Data-500K%20observations-orange)
![Score](https://img.shields.io/badge/Correlation-Validated-green)
![ML](https://img.shields.io/badge/ML-Random%20Forest-red)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)

---

## Problématique

> Comment les conditions météorologiques et le contexte socio-économique influencent-ils
> la fréquence et la nature des crimes à Chicago — et peut-on prédire si un crime donné
> aboutira à une arrestation ?

---

## Analyses exploratoires réalisées

| Section | Description |
|---------|-------------|
| **Portrait de la ville** | Top crimes, évolution 2001–2025 |
| **Saisonnalité** | Juillet–Août = +30% de crimes vs Février |
| **Température vs crimes** | Corrélation validée — seuil critique à 5°C |
| **Pluie comme bouclier** | -6% de crimes les jours pluvieux |
| **Inégalités** | Quartiers défavorisés : 2× plus de crimes par grande chaleur |
| **Comparaison littérature** | Validation vs Ranson (2014) |

---

## Modélisation — Prédiction d'arrestation

**Objectif :** prédire si un crime donné aboutira à une arrestation (`Arrest`), à partir du type de crime, du moment, du lieu et du contexte météo/socio-économique.

| Modèle | Accuracy | Précision (Arrestation) | Rappel (Arrestation) |
|---|---|---|---|
| Dummy (classe majoritaire) | 74.9% | — | — |
| Random Forest | 86.8% | 0.93 | 0.51 |
| Random Forest (`class_weight="balanced"`) | 85.0% | 0.74 | **0.61** |

**Résultat clé :** le type de crime (`Primary Type`) domine très largement l'importance des variables (≈70%), loin devant l'heure, la météo ou le profil socio-économique du quartier (chacun entre 2 et 6%). Le modèle final retenu privilégie le rappel (`class_weight="balanced"`) — un choix assumé pour limiter les arrestations manquées, au prix de plus de faux positifs. Le détail du compromis précision/rappel est discuté dans le notebook.

---

## Dashboard interactif

Une application Streamlit à trois onglets :
- **Predict** — estime la probabilité d'arrestation pour un incident fictif (type de crime, heure, météo, quartier)
- **City Insights** — visualisations clés de l'analyse (top crimes, taux d'arrestation par type, pattern horaire/mensuel, effet température, effet socio-économique par quartier)
- **About** — contexte du projet, sources de données, limites

### Exemples à tester (onglet Predict)

| Scénario | Type de crime | Heure | Remarque |
|---|---|---|---|
| Arrestation quasi automatique | NARCOTICS | 22h | Possession souvent constatée sur le fait |
| Rarement élucidé sur le moment | THEFT | 15h | Découvert après coup, pas de suspect immédiat |
| Crime violent, arrestation attendue | WEAPONS VIOLATION | 2h | Comparer nuit vs jour |
| Isoler l'effet météo | BATTERY | 14h | Comparer -20°C vs 35°C, tout le reste identique |
| Isoler l'effet du quartier | THEFT | 15h | Comparer un quartier à Hardship Index élevé vs faible |

**Conseil :** compare les probabilités relatives entre ces scénarios plutôt que de lire chacun isolément — c'est cette cohérence relative qui révèle si le modèle a appris quelque chose de sensé.

---

## Résultats clés de l'analyse exploratoire

- **+1.06% de crimes violents** par degré Celsius supplémentaire
- **Seuil critique à 5°C** — en dessous, Chicago "hiberne criminellement"
- **La pluie réduit les crimes de 6%**, même par grande chaleur
- Les quartiers défavorisés subissent un effet météo **2× plus fort**
- Résultats cohérents avec **Ranson (2014)** — étude sur 2 997 comtés

---

## Structure du projet

```
.
├── data/
│   ├── chicago_crimes_500k.csv
│   ├── chicago_weather.csv
│   └── Census_Data_Chicago.csv
├── models/
│   ├── model_arrest_rf.pkl
│   └── encoder_crime_type.pkl
├── notebooks/
│   └── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://gitlab.com/Carlos-H-24/Chicago-Crime-Weather-Analysis
cd Chicago-Crime-Weather-Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lancer le notebook d'analyse
jupyter notebook notebooks/analysis.ipynb

# Ou lancer le dashboard interactif
streamlit run app.py
```

---

## Limites et pistes d'amélioration

- Le modèle reflète les crimes **rapportés et enregistrés**, pas la criminalité réelle
- Les données d'arrestation peuvent refléter des biais systémiques dans les pratiques policières, pas uniquement la gravité du crime
- Échantillon de 500 000 lignes tiré d'un dataset bien plus large
- Le déséquilibre de classes (25% d'arrestations) limite le rappel du modèle même après rééquilibrage

---

## Auteur

**AKODJENOU Hervé Carlos**
- GitLab : [Carlos-H-24](https://gitlab.com/Carlos-H-24)
- Email : carlosakodjenou@gmail.com
