"""Forecast contract consumed by the NoShow project.

HotelForecast answers "how many rooms will this hotel sell, and how much demand will it turn
away?". NoShow answers "who among those guests will actually come, and how many rooms should
be oversold?". Overbooking is only worth it where unmet demand exists: this file carries it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hotelforecast.features import MAX_HORIZON, FeatureBuilder
from hotelforecast.models import lgbm
from hotelforecast.simulate import World

CONTRACT_VERSION = "1"
CONTRACT_COLUMNS = [
    "contract_version", "hotel_id", "asof_date", "stay_date", "horizon", "capacity", "otb_rooms",
    "forecast_rooms", "p10_rooms", "p90_rooms", "forecast_adr", "forecast_unmet_rooms",
    "sellout_pressure",
]


def forecast_from(world: World, origin: int) -> pd.DataFrame:
    """Train on everything realised at `origin`, then forecast J+1 to J+90."""
    builder = FeatureBuilder(world)
    model = lgbm.DemandModel().fit(lgbm.training_rows(world, builder, origin))
    rows = builder.rows(origin, np.arange(1, MAX_HORIZON + 1))
    frame = pd.concat([rows, model.predict(rows)], axis=1)
    dates = world.calendar["date"].to_numpy()
    frame["hotel_id"] = world.hotels["hotel_id"].to_numpy()[frame["hotel_idx"].to_numpy()]
    frame["asof_date"] = dates[origin]
    frame["stay_date"] = dates[frame["stay_idx"].to_numpy()]
    return frame


def rolling_forecast(world: World, first: int, last: int, horizon: int) -> pd.DataFrame:
    """Forecast each stay in [first, last] from `horizon` days before it, with a frozen model.

    The model only trains on stays realised before the first origin, so a whole test period
    can be replayed without any future information.
    """
    builder = FeatureBuilder(world)
    model = lgbm.DemandModel().fit(lgbm.training_rows(world, builder, first - horizon))
    rows = pd.concat(
        [builder.rows(stay - horizon, np.array([horizon])) for stay in range(first, last + 1)],
        ignore_index=True,
    )
    frame = pd.concat([rows, model.predict(rows)], axis=1)
    dates = world.calendar["date"].to_numpy()
    frame["hotel_id"] = world.hotels["hotel_id"].to_numpy()[frame["hotel_idx"].to_numpy()]
    frame["asof_date"] = dates[frame["origin_idx"].to_numpy()]
    frame["stay_date"] = dates[frame["stay_idx"].to_numpy()]
    return frame


def to_contract(forecast: pd.DataFrame) -> pd.DataFrame:
    frame = forecast.copy()
    frame["contract_version"] = CONTRACT_VERSION
    frame["sellout_pressure"] = (frame["p90_rooms"] / frame["capacity"]).round(4)
    frame["capacity"] = frame["capacity"].astype(int)
    frame["otb_rooms"] = frame["otb_rooms"].astype(int)
    for column in ("forecast_rooms", "p10_rooms", "p90_rooms", "forecast_unmet_rooms"):
        frame[column] = frame[column].round(2)
    frame["forecast_adr"] = frame["forecast_adr"].round(2)
    return frame[CONTRACT_COLUMNS]


def write_contract(contract: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contract.to_parquet(path, index=False)
