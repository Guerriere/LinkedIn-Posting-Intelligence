# 📊 LinkedIn Posting Intelligence

**LinkedIn Posting Intelligence** est un outil d'analyse de données conçu pour transformer les exports bruts de LinkedIn en insights stratégiques. Ce dashboard permet de visualiser l'engagement, d'identifier les types de contenus les plus performants et d'optimiser sa stratégie de personal branding.

---

## 🚀 Fonctionnalités
* **Visualisation d'Engagement :** Analyse des likes, commentaires et reposts sur une période donnée (2024-2026).
* **Segmentation par Catégorie :** Comparaison de la performance entre les posts de type *Inspiration*, *Storytelling*, *Tips Techniques*, etc.
* **Filtres Dynamiques :** Filtrage par année, type de post (quote, original, etc.) et métriques spécifiques.
* **Calculs Statistiques :** Affichage automatique de la moyenne et de la médiane des réactions pour éviter les biais des posts "viraux" isolés.

## 🛠️ Stack Technique
* **Langage :** Python 3.13
* **Data Processing :** Pandas (Nettoyage de données, gestion des séries temporelles).
* **Interface :** Streamlit (Framework UI rapide pour la Data Science).
* **Déploiement :** Streamlit Cloud.

## 📂 Structure du Projet
```text
├── linkedin_dashboard_v2.py  # Script principal de l'application
├── requirements.txt           # Dépendances du projet
├── data/                      # Données brutes (CSV)
└── README.md                  # Documentation
