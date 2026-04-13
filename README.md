# 🏙️ Chicago Crime & Weather Analysis

> Analyse de l'influence des conditions météorologiques sur la 
> criminalité à Chicago — 500 000 observations sur 20 ans.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Data](https://img.shields.io/badge/Data-500K%20observations-orange)
![Score](https://img.shields.io/badge/Correlation-Validated-green)

---

## Problématique

> **Comment les conditions météorologiques influencent-elles
> la fréquence et la nature des crimes à Chicago ?**

---

## Analyses réalisées

| Section | Description |
|---------|-------------|
| **Portrait de la ville** | Top crimes, évolution 2001–2025 |
| **Saisonnalité** | Juillet–Août = +30% de crimes vs Février |
| **Température vs crimes** | Corrélation validée — seuil critique à 5°C |
| **Pluie comme bouclier** | -6% de crimes les jours pluvieux |
| **Inégalités** | Quartiers défavorisés : 2× plus de crimes par grande chaleur |
| **Comparaison littérature** | Validation vs Ranson (2014) |

---

## Résultats clés

- **+1.06% de crimes violents** par degré Celsius supplémentaire
- **Seuil critique à 5°C** — en dessous Chicago "hiberne criminellement"
- **La pluie réduit les crimes de 6%** même par grande chaleur
- Les quartiers défavorisés subissent un effet météo **2× plus fort**
- Résultats cohérents avec **Ranson (2014)** — étude sur 2 997 comtés

---

## Installation

```bash
git clone https://github.com/Carlos-H-24/Chicago-Crime-Weather-Analysis
cd Chicago-Crime-Weather-Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

---

## Auteurs

**AKODJENOU Hervé Carlos**

- GitHub : [Carlos-H-24](https://github.com/Carlos-H-24)
- Email : carlosakodjenou@gmail.com