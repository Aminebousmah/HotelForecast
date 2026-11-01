"""Local statistical models, fitted per series on actuals known at the origin."""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS


def fit_predict(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast `y` for each `unique_id`. Columns of history: unique_id, ds, y."""
    engine = StatsForecast(
        models=[
            AutoETS(season_length=7, alias="ets"),
            MSTL(season_length=[7, 364], trend_forecaster=AutoETS(model="ZZN"), alias="mstl"),
        ],
        freq="D",
        n_jobs=-1,
    )
    forecast = engine.forecast(df=history, h=horizon)
    return forecast.reset_index(drop="unique_id" in forecast.columns)
