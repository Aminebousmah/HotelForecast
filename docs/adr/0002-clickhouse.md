# ADR 0002 : ClickHouse pour le modèle en étoile

## Contexte
Le backtest produit une ligne par hôtel, nuit, origine et horizon. Le suivi de précision agrège ces lignes dans tous les sens : horizon, marque, région.

## Options
- **PostgreSQL** : universel, mais moins adapté aux agrégations sur de gros volumes de prévisions.
- **DuckDB** : excellent en local, mais sans service partagé pour un outil de BI.
- **ClickHouse** : stockage en colonnes, très rapide en agrégation, service accessible depuis Power BI.

## Décision
ClickHouse en docker, transformé par dbt (`fact_daily_hotel`, `fact_forecast`, `fact_accuracy`).

## Conséquences
- Le dashboard Streamlit lit les fichiers Parquet : il reste utilisable sans lancer docker.
- Un service de plus à faire tourner pour la partie BI, ce qui est acceptable parce qu'il reste optionnel pour le calcul des prévisions.
