# ADR 0001 : un LightGBM global plutôt que des modèles locaux

## Contexte
Il faut prévoir des dizaines de séries hôtel par hôtel, de J+1 à J+90. La donnée la plus informative à court terme, le portefeuille de réservations à la date de prévision, ne rentre pas naturellement dans un ETS ou un ARIMA.

## Options
- **Modèles locaux** (AutoETS, MSTL) : un modèle par hôtel, simple à expliquer, mais sans portefeuille ni partage d'information entre hôtels.
- **LightGBM global** : un seul modèle, une ligne par couple hôtel × horizon, avec portefeuille, pick-up, Y-1, Y-2 et calendrier.

## Décision
LightGBM global en modèle principal. AutoETS et MSTL restent dans le backtest comme points de comparaison.

## Conséquences
- Le modèle exploite le portefeuille : à J+30, il est nettement plus précis que les modèles locaux (voir `reports/accuracy_by_horizon.csv`).
- Le jeu d'entraînement doit être construit avec soin pour éviter toute fuite : chaque ligne ne lit que ce qui est connu à son origine, ce que vérifie `test_no_future_information_leaks_into_features`.
- Un nouvel hôtel profite tout de suite de ce que le modèle a appris sur les autres.
