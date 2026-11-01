"""Reference forecasts every model has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hotelforecast.features import ALIGNED_YEAR
from hotelforecast.simulate import World

RMS_SEED = 2024


def naive_y1(world: World, rows: pd.DataFrame) -> np.ndarray:
    """Rooms sold on the aligned weekday one year earlier."""
    stays = rows["stay_idx"].to_numpy() - ALIGNED_YEAR
    return world.rooms_sold[rows["hotel_idx"].to_numpy(), stays].astype(float)


def moving_average(world: World, rows: pd.DataFrame, window: int = 28) -> np.ndarray:
    """Mean rooms sold over the `window` days ending at the origin."""
    cumulative = np.cumsum(world.rooms_sold, axis=1)
    origins = rows["origin_idx"].to_numpy()
    hotels = rows["hotel_idx"].to_numpy()
    return (cumulative[hotels, origins] - cumulative[hotels, origins - window]) / window


def rms_forecast(world: World, rows: pd.DataFrame) -> np.ndarray:
    """Simulated RMS: additive pick-up, blind to the event calendar, with a stable per-hotel bias.

    It reads on-the-books like a real RMS, so it reacts to events only once bookings arrive.
    """
    rng = np.random.default_rng(RMS_SEED)
    hotel_bias = np.asarray(rng.normal(0, 0.025, size=world.rooms_sold.shape[0]))
    hotels = rows["hotel_idx"].to_numpy()
    capacity = rows["capacity"].to_numpy()
    raw = rows["otb_rooms"].to_numpy() + rows["pickup_ratio"].to_numpy() * capacity
    return np.clip(raw * (1 + hotel_bias[hotels]), 0, capacity)
