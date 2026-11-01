"""Point-in-time features. Everything here only reads what is known at the forecast origin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hotelforecast.simulate import BRANDS, REGIONS, World, event_multipliers

ALIGNED_YEAR = 364  # keeps the day of week, unlike a calendar year
MAX_HORIZON = 90
FEATURE_COLUMNS = [
    "horizon", "dow", "month", "doy_sin", "doy_cos", "is_public_holiday", "school_holiday_share",
    "event_multiplier", "brand_code", "region_code", "capacity", "otb_ratio", "pace_7d_ratio",
    "pickup_ratio", "y1_ratio", "y2_ratio", "otb_y1_ratio", "pace_vs_y1_ratio", "recent_28d_ratio",
    "recent_dow_ratio", "adr_recent", "adr_y1",
]


def aligned_lag(values: np.ndarray, days: int) -> np.ndarray:
    """Shift a daily array by `days` along its last axis, padding the start with NaN."""
    result = np.full(values.shape, np.nan, dtype=float)
    if days < values.shape[-1]:
        result[..., days:] = values[..., : values.shape[-1] - days]
    return result


def pickup_estimate(
    otb: np.ndarray, sold: np.ndarray, stay: int, horizon: int, origin: int, n_weeks: int = 8
) -> np.ndarray:
    """Average rooms picked up between lead `horizon` and arrival on past same-weekday stays."""
    first_lag = max(1, int(np.ceil((stay - origin) / 7)))
    past = stay - 7 * np.arange(first_lag, first_lag + n_weeks)
    past = past[past >= 0]
    if past.size == 0:
        return np.zeros(otb.shape[0])
    return (sold[:, past] - otb[:, past, horizon]).mean(axis=1)


class FeatureBuilder:
    """Builds one row per (hotel, origin, horizon), reading only data available at the origin."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.calendar = world.calendar
        self.capacity = world.hotels["capacity"].to_numpy(dtype=float)
        self.brand_code = world.hotels["brand"].map({b: i for i, b in enumerate(BRANDS)}).to_numpy()
        self.region_code = (
            world.hotels["region"].map({r: i for i, r in enumerate(REGIONS)}).to_numpy()
        )
        events = event_multipliers(self.calendar)
        self.events = np.stack([events[r] for r in world.hotels["region"]])
        self.adr = world.adr

    def rows(self, origin: int, horizons: np.ndarray) -> pd.DataFrame:
        """Feature rows for one origin day index and an array of horizons."""
        frames = [self._one_horizon(origin, int(h)) for h in horizons]
        return pd.concat(frames, ignore_index=True)

    def _one_horizon(self, origin: int, h: int) -> pd.DataFrame:
        world, capacity = self.world, self.capacity
        stay = origin + h
        sold, otb = world.rooms_sold, world.otb
        n_hotels = len(capacity)
        row = self.calendar.iloc[stay]

        otb_now = otb[:, stay, h].astype(float)
        otb_7d = otb[:, stay, h + 7].astype(float)
        y1 = stay - ALIGNED_YEAR
        y2 = stay - 2 * ALIGNED_YEAR
        y1_sold = sold[:, y1].astype(float) if y1 >= 0 else np.full(n_hotels, np.nan)
        y2_sold = sold[:, y2].astype(float) if y2 >= 0 else np.full(n_hotels, np.nan)
        otb_y1 = otb[:, y1, h].astype(float) if y1 >= 0 else np.full(n_hotels, np.nan)
        recent = sold[:, max(0, origin - 27) : origin + 1].mean(axis=1)
        same_dow = stay - 7 * np.arange(int(np.ceil(h / 7)), int(np.ceil(h / 7)) + 4)
        same_dow = same_dow[same_dow >= 0]
        recent_dow = sold[:, same_dow].mean(axis=1) if same_dow.size else recent
        adr_window = self.adr[:, max(0, origin - 27) : origin + 1]

        return pd.DataFrame(
            {
                "hotel_idx": np.arange(n_hotels),
                "origin_idx": origin,
                "stay_idx": stay,
                "horizon": h,
                "dow": row["dow"],
                "month": row["month"],
                "doy_sin": np.sin(2 * np.pi * row["doy"] / 365.25),
                "doy_cos": np.cos(2 * np.pi * row["doy"] / 365.25),
                "is_public_holiday": int(row["is_public_holiday"]),
                "school_holiday_share": row["school_holiday_share"],
                "event_multiplier": self.events[:, stay],
                "brand_code": self.brand_code,
                "region_code": self.region_code,
                "capacity": capacity,
                "otb_rooms": otb_now,
                "otb_ratio": otb_now / capacity,
                "pace_7d_ratio": (otb_now - otb_7d) / capacity,
                "pickup_ratio": pickup_estimate(otb, sold, stay, h, origin) / capacity,
                "y1_rooms": y1_sold,
                "y1_ratio": y1_sold / capacity,
                "y2_ratio": y2_sold / capacity,
                "otb_y1_ratio": otb_y1 / capacity,
                "pace_vs_y1_ratio": (otb_now - otb_y1) / capacity,
                "recent_28d_ratio": recent / capacity,
                "recent_dow_ratio": recent_dow / capacity,
                "adr_recent": adr_window.mean(axis=1),
                "adr_y1": self.adr[:, y1] if y1 >= 0 else np.full(n_hotels, np.nan),
            }
        )
