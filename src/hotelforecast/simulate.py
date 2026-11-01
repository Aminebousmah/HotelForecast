"""Deterministic generator of a fictional hotel group.

Every number downstream derives from a single reservation log: booking requests are
simulated one by one, accepted while inventory is available, and possibly cancelled later.
Denied requests are kept, with their latent outcome, so that the NoShow project can replay
overbooking decisions against a true counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates

MAX_LEAD = 120
BRANDS = ("economy", "midscale", "upscale")
REGIONS = ("idf", "paca", "aura", "occitanie", "grand_est")
SEGMENTS = ("ota", "direct", "corporate", "group", "offline_ta")
DEPOSITS = ("none", "refundable", "non_refundable")
CUSTOMER_TYPES = ("transient", "contract", "group")

BRAND_PARAMS: dict[str, dict[str, Any]] = {
    "economy": {"capacity": (55, 95), "price": 72.0, "weight": 0.50},
    "midscale": {"capacity": (80, 135), "price": 108.0, "weight": 0.35},
    "upscale": {"capacity": (120, 210), "price": 185.0, "weight": 0.15},
}
SEGMENT_MIX = {
    "economy": (0.48, 0.24, 0.14, 0.06, 0.08),
    "midscale": (0.40, 0.20, 0.22, 0.10, 0.08),
    "upscale": (0.32, 0.22, 0.24, 0.14, 0.08),
}
SEGMENT_LEAD_MEAN = {"ota": 26, "direct": 18, "corporate": 9, "group": 55, "offline_ta": 40}
SEGMENT_PRICE = {"ota": 1.0, "direct": 0.97, "corporate": 0.9, "group": 0.8, "offline_ta": 0.86}
SEGMENT_LOGIT = {"ota": 0.55, "direct": -0.15, "corporate": -0.55, "group": -0.25, "offline_ta": 0}

# (region, month, day, duration in days, demand multiplier): recurring fictional trade fairs.
RECURRING_EVENTS = (
    ("idf", 1, 20, 4, 1.25),
    ("idf", 3, 12, 5, 1.30),
    ("idf", 10, 6, 4, 1.28),
    ("aura", 1, 24, 4, 1.35),
    ("paca", 5, 14, 11, 1.45),
    ("grand_est", 12, 8, 14, 1.30),
    ("occitanie", 9, 18, 3, 1.25),
)
OLYMPICS = (pd.Timestamp("2024-07-26"), pd.Timestamp("2024-08-11"))


@dataclass(frozen=True)
class SimConfig:
    n_hotels: int = 60
    start: str = "2022-01-01"
    years: int = 4
    seed: int = 42


@dataclass
class World:
    """Simulated group. Hotel-by-day arrays share the (hotel, day) index order."""

    hotels: pd.DataFrame
    calendar: pd.DataFrame
    reservations: pd.DataFrame
    otb: np.ndarray  # (hotel, stay day, days before stay) -> rooms on the books
    rooms_sold: np.ndarray
    rooms_occupied: np.ndarray
    room_revenue: np.ndarray
    rooms_denied: np.ndarray

    @property
    def adr(self) -> np.ndarray:
        return np.divide(
            self.room_revenue, self.rooms_sold, out=np.zeros_like(self.room_revenue),
            where=self.rooms_sold > 0,
        )


def make_hotels(n_hotels: int, rng: np.random.Generator) -> pd.DataFrame:
    weights = [BRAND_PARAMS[b]["weight"] for b in BRANDS]
    brands = rng.choice(np.array(BRANDS), size=n_hotels, p=weights)
    regions = rng.choice(np.array(REGIONS), size=n_hotels, p=(0.34, 0.2, 0.18, 0.14, 0.14))
    capacity = [int(rng.integers(*BRAND_PARAMS[b]["capacity"])) for b in brands]
    business_share = rng.beta(2.2, 2.2, size=n_hotels)
    # A few hotels lose demand steadily: the drift report has to find them.
    drift_ids = rng.choice(n_hotels, size=max(1, n_hotels // 15), replace=False)
    is_drifting = np.isin(np.arange(n_hotels), drift_ids)
    return pd.DataFrame(
        {
            "hotel_id": [f"H{i + 1:04d}" for i in range(n_hotels)],
            "brand": brands,
            "region": regions,
            "capacity": capacity,
            "business_share": business_share.round(3),
            "base_load": rng.uniform(0.72, 1.08, size=n_hotels).round(3),
            "price_index": rng.uniform(0.85, 1.25, size=n_hotels).round(3),
            "is_drifting": is_drifting,
        }
    )


def make_calendar(start: str, years: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=int(round(365.25 * years)), freq="D")
    public = holidays.country_holidays("FR", years=sorted(set(dates.year)))
    school = SchoolHolidayDates()
    school_days: dict = {}
    for year in sorted(set(dates.year)):
        school_days.update(school.holidays_for_year(year))
    share = []
    for d in dates:
        info = school_days.get(d.date())
        share.append(
            0.0 if info is None
            else (info["vacances_zone_a"] + info["vacances_zone_b"] + info["vacances_zone_c"]) / 3
        )
    return pd.DataFrame(
        {
            "date": dates,
            "dow": dates.day_of_week,
            "month": dates.month,
            "doy": dates.day_of_year,
            "is_public_holiday": [d.date() in public for d in dates],
            "school_holiday_share": np.round(share, 3),
        }
    )


def event_multipliers(calendar: pd.DataFrame) -> dict[str, np.ndarray]:
    """Known demand multiplier per region and day: trade fairs and the 2024 Olympics."""
    dates = pd.DatetimeIndex(calendar["date"])
    multipliers = {region: np.ones(len(dates)) for region in REGIONS}
    for region, month, day, duration, factor in RECURRING_EVENTS:
        for year in sorted(set(dates.year)):
            start = pd.Timestamp(year=year, month=month, day=day)
            mask = (dates >= start) & (dates < start + pd.Timedelta(days=duration))
            multipliers[region][mask] *= factor
    during = (dates >= OLYMPICS[0]) & (dates <= OLYMPICS[1])
    before = (dates >= OLYMPICS[0] - pd.Timedelta(days=21)) & (dates < OLYMPICS[0])
    multipliers["idf"][during] *= 1.35
    multipliers["idf"][before] *= 0.82
    return multipliers


def _seasonality(calendar: pd.DataFrame, hotel: pd.Series) -> np.ndarray:
    doy, dow = calendar["doy"].to_numpy(), calendar["dow"].to_numpy()
    business = float(hotel["business_share"])
    leisure_wave = 1 + 0.28 * np.cos(2 * np.pi * (doy - 205) / 365.25)
    business_wave = 1 + 0.14 * np.cos(2 * np.pi * (doy - 280) / 182.6)
    weekly_business = np.array([1.08, 1.22, 1.24, 1.14, 0.84, 0.56, 0.72])[dow]
    weekly_leisure = np.array([0.82, 0.8, 0.84, 0.92, 1.12, 1.28, 0.96])[dow]
    season = business * business_wave * weekly_business + (1 - business) * leisure_wave * (
        weekly_leisure * (1 + 0.22 * calendar["school_holiday_share"].to_numpy())
    )
    public = calendar["is_public_holiday"].to_numpy()
    return season * np.where(public, 0.92 + 0.3 * (1 - business), 1.0)


def _demand_curve(
    calendar: pd.DataFrame, hotel: pd.Series, events: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Expected room-nights demanded per stay day, as a multiple of capacity."""
    n_days = len(calendar)
    noise = np.empty(n_days)
    noise[0] = 0.0
    shocks = rng.normal(0, 0.055, size=n_days)
    for t in range(1, n_days):
        noise[t] = 0.82 * noise[t - 1] + shocks[t]
    curve = float(hotel["base_load"]) * _seasonality(calendar, hotel) * events * np.exp(noise)
    if rng.random() < 0.15:  # temporary renovation: part of the inventory demand disappears
        start = int(rng.integers(90, n_days - 60))
        curve[start : start + int(rng.integers(21, 56))] *= 0.62
    if hotel["is_drifting"]:
        start = int(n_days * 0.82)
        curve[start:] *= np.linspace(1.0, 0.68, n_days - start)
    return curve


def _sample_requests(
    hotel: pd.Series, curve: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw every booking request for one hotel, with its latent outcome."""
    n_days = len(curve)
    capacity = int(hotel["capacity"])
    mean_los = 1.75
    arrivals = rng.poisson(curve * capacity / mean_los)
    arrival_idx = np.repeat(np.arange(n_days), arrivals)
    n = len(arrival_idx)

    mix = SEGMENT_MIX[hotel["brand"]]
    segment_idx = rng.choice(len(SEGMENTS), size=n, p=mix)
    segments = np.array(SEGMENTS)[segment_idx]
    lead_mean = np.array([SEGMENT_LEAD_MEAN[s] for s in SEGMENTS])[segment_idx]
    lead = np.minimum(rng.gamma(1.15, lead_mean / 1.15).astype(int), MAX_LEAD)
    los = np.minimum(1 + rng.poisson(0.75, size=n), 10)

    deposit_probs = np.where(segment_idx == 1, 0.22, np.where(segment_idx == 3, 0.5, 0.12))
    u = rng.random(n)
    deposit_idx = np.where(u < deposit_probs, 2, np.where(u < deposit_probs + 0.2, 1, 0))
    repeated = rng.random(n) < np.where(segment_idx == 2, 0.28, 0.05)
    prev_cancel = rng.poisson(0.05, size=n) + rng.poisson(0.12, size=n) * repeated
    prev_ok = rng.poisson(0.05, size=n) + rng.poisson(1.8, size=n) * repeated
    requests = np.minimum(rng.poisson(0.55, size=n), 5)
    parking = rng.random(n) < 0.06
    adults = np.clip(rng.poisson(0.95, size=n) + 1, 1, 4)
    children = rng.poisson(0.1, size=n)
    customer_idx = np.where(segment_idx == 3, 2, np.where(segment_idx == 2, 1, 0))

    pressure = np.clip(curve[arrival_idx], 0.3, 1.5)
    base_price = BRAND_PARAMS[hotel["brand"]]["price"] * float(hotel["price_index"])
    channel_price = np.array([SEGMENT_PRICE[s] for s in SEGMENTS])[segment_idx]
    adr = base_price * channel_price * (0.72 + 0.34 * pressure) * rng.lognormal(0, 0.06, size=n)

    latent = rng.normal(0, 0.55, size=n)  # what no booking system ever records
    logit = (
        -1.42
        + np.array([SEGMENT_LOGIT[s] for s in SEGMENTS])[segment_idx]
        + 1.15 * np.minimum(lead, 120) / 120
        + np.array([0.0, -0.35, -2.1])[deposit_idx]
        - 0.95 * repeated
        + 0.75 * np.minimum(prev_cancel, 2)
        - 0.18 * np.minimum(prev_ok, 5)
        - 0.32 * requests
        - 0.25 * parking
        + 0.55 * (adr / (base_price * (0.72 + 0.34 * pressure)) - 1)
        + latent
    )
    will_cancel = rng.random(n) < 1 / (1 + np.exp(-logit))
    will_cancel &= lead > 0
    cancel_before = np.floor(lead * rng.beta(1.0, 2.6, size=n)).astype(int)
    noshow_logit = (
        -3.05 + 0.5 * (segment_idx == 0) - 1.1 * (deposit_idx == 2) - 0.5 * repeated + 0.3 * latent
    )
    will_noshow = ~will_cancel & (rng.random(n) < 1 / (1 + np.exp(-noshow_logit)))

    return {
        "arrival_idx": arrival_idx,
        "lead_time": lead,
        "los": los,
        "market_segment": segments,
        "deposit_type": np.array(DEPOSITS)[deposit_idx],
        "customer_type": np.array(CUSTOMER_TYPES)[customer_idx],
        "is_repeated_guest": repeated,
        "previous_cancellations": prev_cancel,
        "previous_bookings_not_canceled": prev_ok,
        "total_of_special_requests": requests,
        "required_car_parking_spaces": parking.astype(int),
        "adults": adults,
        "children": children,
        "adr": np.round(adr, 2),
        "will_cancel": will_cancel,
        "will_noshow": will_noshow,
        "cancel_before": cancel_before,
    }


def _allocate(requests: dict[str, np.ndarray], capacity: int, n_days: int) -> np.ndarray:
    """Replay bookings and cancellations chronologically against room inventory."""
    arrival, lead = requests["arrival_idx"], requests["lead_time"]
    los, cancels = requests["los"], requests["will_cancel"]
    cancel_before = requests["cancel_before"]
    book_day = arrival - lead
    cancel_day = arrival - cancel_before

    n = len(arrival)
    # Same-day cancellations release rooms before new bookings are evaluated.
    cancel_ids = np.flatnonzero(cancels)
    event_day = np.concatenate([cancel_day[cancel_ids], book_day])
    event_kind = np.concatenate([np.zeros(len(cancel_ids), int), np.ones(n, int)])
    event_id = np.concatenate([cancel_ids, np.arange(n)])
    order = np.lexsort((event_kind, event_day))

    inventory = [0] * (n_days + 12)
    accepted = np.zeros(n, dtype=bool)
    arrival_list, los_list = arrival.tolist(), los.tolist()
    for kind, rid in zip(event_kind[order].tolist(), event_id[order].tolist(), strict=True):
        first = arrival_list[rid]
        nights = range(first, first + los_list[rid])
        if kind == 0:
            if accepted[rid]:
                for night in nights:
                    inventory[night] -= 1
        elif all(inventory[night] < capacity for night in nights):
            accepted[rid] = True
            for night in nights:
                inventory[night] += 1
    return accepted


def simulate(config: SimConfig | None = None) -> World:
    config = config or SimConfig()
    rng = np.random.default_rng(config.seed)
    hotels = make_hotels(config.n_hotels, rng)
    calendar = make_calendar(config.start, config.years)
    events = event_multipliers(calendar)
    n_hotels, n_days = len(hotels), len(calendar)

    otb = np.zeros((n_hotels, n_days, MAX_LEAD + 1), dtype=np.int16)
    sold = np.zeros((n_hotels, n_days), dtype=np.int32)
    occupied = np.zeros((n_hotels, n_days), dtype=np.int32)
    revenue = np.zeros((n_hotels, n_days), dtype=np.float64)
    denied = np.zeros((n_hotels, n_days), dtype=np.int32)
    frames: list[pd.DataFrame] = []
    dates = pd.DatetimeIndex(calendar["date"])

    for h, hotel in hotels.iterrows():
        h = int(h)  # type: ignore[arg-type]
        curve = _demand_curve(calendar, hotel, events[hotel["region"]], rng)
        requests = _sample_requests(hotel, curve, rng)
        accepted = _allocate(requests, int(hotel["capacity"]), n_days)
        _accumulate(h, requests, accepted, n_days, otb, sold, occupied, revenue, denied)

        arrival = requests.pop("arrival_idx")
        cancel_before = requests.pop("cancel_before")
        frame = pd.DataFrame(requests)
        frame.insert(0, "hotel_id", hotel["hotel_id"])
        frame.insert(1, "arrival_date", dates[arrival])
        frame.insert(2, "booking_date", dates[arrival] - pd.to_timedelta(frame["lead_time"], "D"))
        frame["cancel_date"] = (dates[arrival] - pd.to_timedelta(cancel_before, "D")).where(
            frame["will_cancel"]
        )
        frame["accepted"] = accepted
        frames.append(frame)

    reservations = pd.concat(frames, ignore_index=True)
    reservations.insert(0, "reservation_id", np.arange(len(reservations), dtype=np.int64))
    return World(hotels, calendar, reservations, otb, sold, occupied, revenue, denied)


def _accumulate(
    h: int,
    requests: dict[str, np.ndarray],
    accepted: np.ndarray,
    n_days: int,
    otb: np.ndarray,
    sold: np.ndarray,
    occupied: np.ndarray,
    revenue: np.ndarray,
    denied: np.ndarray,
) -> None:
    """Expand reservations to stay nights and fill the hotel-by-day arrays."""
    los = requests["los"]
    rid = np.repeat(np.arange(len(los)), los)
    offset = np.concatenate([np.arange(n) for n in los]) if len(los) else np.array([], int)
    night = requests["arrival_idx"][rid] + offset
    inside = night < n_days
    rid, offset, night = rid[inside], offset[inside], night[inside]

    refused = ~accepted[rid]
    np.add.at(denied[h], night[refused], 1)

    keep = ~refused
    rid, offset, night = rid[keep], offset[keep], night[keep]
    # Days before each stay night at which the reservation enters and, if it does, leaves.
    enter = np.minimum(requests["lead_time"][rid] + offset, MAX_LEAD)
    delta = np.zeros((n_days, MAX_LEAD + 2), dtype=np.int32)
    np.add.at(delta, (night, enter), 1)
    cancels = requests["will_cancel"][rid]
    leave = np.minimum(requests["cancel_before"][rid] + offset, MAX_LEAD)
    np.add.at(delta, (night[cancels], leave[cancels]), -1)
    # Rooms on the books at lead L: everything entered at lead >= L, minus what already left.
    otb[h] = np.cumsum(delta[:, ::-1], axis=1)[:, ::-1][:, : MAX_LEAD + 1].astype(np.int16)

    stays = ~cancels
    np.add.at(sold[h], night[stays], 1)
    shows = stays & ~requests["will_noshow"][rid]
    np.add.at(occupied[h], night[shows], 1)
    np.add.at(revenue[h], night[stays], requests["adr"][rid][stays])


def save_world(world: World, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    world.hotels.to_parquet(out_dir / "dim_hotel.parquet", index=False)
    world.calendar.to_parquet(out_dir / "dim_date.parquet", index=False)
    world.reservations.to_parquet(out_dir / "reservations.parquet", index=False)
    np.savez_compressed(
        out_dir / "arrays.npz",
        otb=world.otb,
        rooms_sold=world.rooms_sold,
        rooms_occupied=world.rooms_occupied,
        room_revenue=world.room_revenue,
        rooms_denied=world.rooms_denied,
    )


def load_world(data_dir: Path) -> World:
    arrays = np.load(data_dir / "arrays.npz")
    return World(
        hotels=pd.read_parquet(data_dir / "dim_hotel.parquet"),
        calendar=pd.read_parquet(data_dir / "dim_date.parquet"),
        reservations=pd.read_parquet(data_dir / "reservations.parquet"),
        otb=arrays["otb"],
        rooms_sold=arrays["rooms_sold"],
        rooms_occupied=arrays["rooms_occupied"],
        room_revenue=arrays["room_revenue"],
        rooms_denied=arrays["rooms_denied"],
    )


def daily_fact(world: World) -> pd.DataFrame:
    """Long hotel-by-day table with the occupancy, ADR and RevPAR KPIs."""
    n_hotels, n_days = world.rooms_sold.shape
    capacity = world.hotels["capacity"].to_numpy()
    fact = pd.DataFrame(
        {
            "hotel_id": np.repeat(world.hotels["hotel_id"].to_numpy(), n_days),
            "stay_date": np.tile(world.calendar["date"].to_numpy(), n_hotels),
            "capacity": np.repeat(capacity, n_days),
            "rooms_sold": world.rooms_sold.ravel(),
            "rooms_occupied": world.rooms_occupied.ravel(),
            "rooms_denied": world.rooms_denied.ravel(),
            "room_revenue": world.room_revenue.ravel().round(2),
        }
    )
    fact["occupancy"] = (fact["rooms_sold"] / fact["capacity"]).round(4)
    sold = fact["rooms_sold"].where(fact["rooms_sold"] > 0)
    fact["adr"] = (fact["room_revenue"] / sold).round(2)
    fact["revpar"] = (fact["room_revenue"] / fact["capacity"]).round(2)
    return fact
