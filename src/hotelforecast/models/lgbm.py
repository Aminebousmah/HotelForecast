"""Global multi-series LightGBM, trained on (hotel, origin, horizon) rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from hotelforecast.features import FEATURE_COLUMNS, MAX_HORIZON, FeatureBuilder
from hotelforecast.simulate import World

PARAMS: dict[str, Any] = {
    "n_estimators": 260,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
CATEGORICAL = ["dow", "month", "brand_code", "region_code"]


def training_rows(
    world: World, builder: FeatureBuilder, cutoff: int, step: int = 5
) -> pd.DataFrame:
    """Rows whose stay is fully realised at `cutoff`, so training never sees the test period."""
    start = 364 + MAX_HORIZON + 28
    frames = []
    for origin in range(start, cutoff, step):
        horizons = np.arange(1, MAX_HORIZON + 1)
        horizons = horizons[origin + horizons <= cutoff]
        if horizons.size:
            frames.append(builder.rows(origin, horizons))
    rows = pd.concat(frames, ignore_index=True)
    hotels, stays = rows["hotel_idx"].to_numpy(), rows["stay_idx"].to_numpy()
    capacity = rows["capacity"].to_numpy()
    rows["target_sold"] = world.rooms_sold[hotels, stays] / capacity
    rows["target_denied"] = world.rooms_denied[hotels, stays] / capacity
    rows["target_adr"] = world.adr[hotels, stays]
    return rows


@dataclass
class DemandModel:
    """Quantile model of rooms sold, plus companion models for ADR and unmet demand."""

    quantiles: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)
    adr: lgb.LGBMRegressor | None = None
    denied: lgb.LGBMRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> DemandModel:
        X = rows[FEATURE_COLUMNS]
        for alpha in (0.1, 0.5, 0.9):
            model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **PARAMS)
            self.quantiles[alpha] = model.fit(
                X, rows["target_sold"], categorical_feature=CATEGORICAL
            )
        sold_mask = rows["target_sold"] > 0
        self.adr = lgb.LGBMRegressor(objective="regression", **PARAMS).fit(
            X[sold_mask], rows.loc[sold_mask, "target_adr"], categorical_feature=CATEGORICAL
        )
        self.denied = lgb.LGBMRegressor(objective="regression", **PARAMS).fit(
            X, rows["target_denied"], categorical_feature=CATEGORICAL
        )
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Forecasts capped at capacity, non-negative, with ordered quantiles."""
        if self.adr is None or self.denied is None:
            raise RuntimeError("fit the model before predicting")
        X = rows[FEATURE_COLUMNS]
        capacity = rows["capacity"].to_numpy()
        stacked = np.column_stack([self.quantiles[a].predict(X) for a in (0.1, 0.5, 0.9)])
        low, mid, high = (np.clip(c * capacity, 0, capacity) for c in np.sort(stacked, axis=1).T)
        # Rooms already on the books can still cancel, so they do not floor the forecast.
        return pd.DataFrame(
            {
                "forecast_rooms": mid,
                "p10_rooms": low,
                "p90_rooms": high,
                "forecast_adr": np.clip(self.adr.predict(X), 0, None),
                "forecast_unmet_rooms": np.clip(self.denied.predict(X) * capacity, 0, None),
            }
        )
