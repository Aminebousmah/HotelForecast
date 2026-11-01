"""Rolling-origin evaluation: each fold trains only on what was known at its origin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hotelforecast import metrics
from hotelforecast.features import MAX_HORIZON, FeatureBuilder
from hotelforecast.models import baseline, lgbm, reconcile, stats
from hotelforecast.simulate import World

MODEL_COLUMNS = {
    "naive_y1": "Naïve Y-1",
    "moving_avg": "Moyenne mobile 28 j",
    "ets": "AutoETS",
    "mstl": "MSTL",
    "rms": "RMS simulé",
    "forecast_rooms": "LightGBM global",
    "reconciled": "LightGBM réconcilié",
}


def fold_origins(n_days: int, n_folds: int) -> list[int]:
    """Origins spaced evenly so that the last fold's horizon ends on the last simulated day."""
    last = n_days - 1 - MAX_HORIZON
    return [last - i * 56 for i in reversed(range(n_folds))]


def _series_history(world: World, origin: int) -> pd.DataFrame:
    dates = world.calendar["date"].to_numpy()[: origin + 1]
    frames = [
        pd.DataFrame({"unique_id": hotel_id, "ds": dates, "y": world.rooms_sold[h, : origin + 1]})
        for h, hotel_id in enumerate(world.hotels["hotel_id"])
    ]
    return pd.concat(frames, ignore_index=True)


def _add_statistical(world: World, frame: pd.DataFrame, origin: int) -> pd.DataFrame:
    predicted = stats.fit_predict(_series_history(world, origin), MAX_HORIZON)
    predicted["stay_date"] = pd.to_datetime(predicted["ds"])
    predicted = predicted.rename(columns={"unique_id": "hotel_id"})
    columns = ["hotel_id", "stay_date", "ets", "mstl"]
    frame = frame.merge(predicted[columns], on=["hotel_id", "stay_date"], how="left")
    capacity = frame["capacity"]
    for column in ("ets", "mstl"):
        frame[column] = frame[column].clip(lower=0).clip(upper=capacity)
    return frame


def _add_reconciled(world: World, frame: pd.DataFrame) -> pd.DataFrame:
    """Reconcile LightGBM at the hotels with ETS above them. Kept as a measured comparison."""
    hotels = world.hotels
    summing, names = reconcile.summing_matrix(hotels)
    n_upper = len(names) - len(hotels)
    bottom_ids = hotels["hotel_id"].tolist()
    capacity = hotels["capacity"].to_numpy(dtype=float)
    lgb_pivot = frame.pivot(index="hotel_id", columns="stay_date", values="forecast_rooms")
    ets_pivot = frame.pivot(index="hotel_id", columns="stay_date", values="ets")
    lgb_matrix = lgb_pivot.loc[bottom_ids].to_numpy()
    ets_matrix = ets_pivot.loc[bottom_ids].to_numpy()
    # Upper nodes carry the smoother ETS signal, hotels carry LightGBM.
    base = np.vstack([summing[:n_upper] @ ets_matrix, lgb_matrix])
    capped, _ = reconcile.cap_and_aggregate(reconcile.wls_struct(base, summing), capacity, summing)
    long = pd.DataFrame(capped, index=lgb_pivot.loc[bottom_ids].index, columns=lgb_pivot.columns)
    long = long.reset_index().melt("hotel_id", var_name="stay_date", value_name="reconciled")
    long["stay_date"] = pd.to_datetime(long["stay_date"])
    return frame.merge(long, on=["hotel_id", "stay_date"], how="left")


def run_fold(world: World, builder: FeatureBuilder, origin: int, fold: int) -> pd.DataFrame:
    train = lgbm.training_rows(world, builder, origin)
    model = lgbm.DemandModel().fit(train)
    rows = builder.rows(origin, np.arange(1, MAX_HORIZON + 1))
    predicted = model.predict(rows)
    frame = pd.concat([rows, predicted], axis=1)

    hotels, stays = frame["hotel_idx"].to_numpy(), frame["stay_idx"].to_numpy()
    dates = world.calendar["date"].to_numpy()
    frame["fold"] = fold
    frame["hotel_id"] = world.hotels["hotel_id"].to_numpy()[hotels]
    frame["brand"] = world.hotels["brand"].to_numpy()[hotels]
    frame["region"] = world.hotels["region"].to_numpy()[hotels]
    frame["origin_date"] = dates[frame["origin_idx"].to_numpy()]
    frame["stay_date"] = dates[stays]
    frame["actual_rooms"] = world.rooms_sold[hotels, stays].astype(float)
    frame["actual_adr"] = world.adr[hotels, stays]
    frame["actual_unmet_rooms"] = world.rooms_denied[hotels, stays].astype(float)
    frame["naive_y1"] = baseline.naive_y1(world, frame)
    frame["moving_avg"] = baseline.moving_average(world, frame)
    frame["rms"] = baseline.rms_forecast(world, frame)
    frame = _add_statistical(world, frame, origin)
    return _add_reconciled(world, frame)


def run_backtest(world: World, n_folds: int = 6) -> pd.DataFrame:
    builder = FeatureBuilder(world)
    origins = fold_origins(world.rooms_sold.shape[1], n_folds)
    return pd.concat(
        [run_fold(world, builder, origin, i + 1) for i, origin in enumerate(origins)],
        ignore_index=True,
    )


def _row(group: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "wape": metrics.wape(group["actual_rooms"], group[column]),
        "bias": metrics.bias(group["actual_rooms"], group[column]),
    }


def accuracy_by_horizon(results: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """WAPE and bias per model. Each horizon H covers the days H-6..H to smooth weekday effects."""
    records = []
    for horizon in horizons:
        window = results[results["horizon"].between(max(1, horizon - 6), horizon)]
        for column, label in MODEL_COLUMNS.items():
            records.append({"horizon": horizon, "model": label, **_row(window, column)})
    return pd.DataFrame(records)


def accuracy_by_segment(results: pd.DataFrame, by: str, horizon: int = 30) -> pd.DataFrame:
    window = results[results["horizon"].between(horizon - 6, horizon)]
    records = []
    for segment, group in window.groupby(by):
        for column in ("naive_y1", "rms", "forecast_rooms"):
            records.append(
                {by: segment, "model": MODEL_COLUMNS[column], **_row(group, column)}
            )
    return pd.DataFrame(records)


def accuracy_by_level(results: pd.DataFrame) -> pd.DataFrame:
    """Does reconciliation pay off? WAPE of raw vs reconciled forecasts at each hierarchy level."""
    levels = {"hôtel": ["hotel_id"], "marque × région": ["brand", "region"], "marque": ["brand"],
              "groupe": []}
    records = []
    for level, keys in levels.items():
        grouped = results.groupby(["fold", "stay_date", "horizon", *keys], observed=True)[
            ["actual_rooms", "forecast_rooms", "reconciled"]
        ].sum()
        for column, label in (("forecast_rooms", "LightGBM global"), ("reconciled", "Réconcilié")):
            records.append(
                {
                    "level": level,
                    "model": label,
                    "wape": metrics.wape(grouped["actual_rooms"], grouped[column]),
                }
            )
    return pd.DataFrame(records)


def error_heatmap(results: pd.DataFrame, max_horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Absolute error rate by month and day of week, up to a maximum horizon."""
    window = results[results["horizon"] <= max_horizon]
    window = window.assign(error=(window["actual_rooms"] - window["forecast_rooms"]).abs())
    grouped = window.groupby(["month", "dow"])[["error", "actual_rooms"]].sum()
    return (grouped["error"] / grouped["actual_rooms"]).unstack("dow")


def interval_coverage(results: pd.DataFrame) -> float:
    return metrics.coverage(results["actual_rooms"], results["p10_rooms"], results["p90_rooms"])


def group_coherence_gap(results: pd.DataFrame, world: World) -> float:
    """Largest gap between the group total and the sum of reconciled hotels."""
    capacity = world.hotels["capacity"].to_numpy(dtype=float)
    summing, _ = reconcile.summing_matrix(world.hotels)
    pivot = (
        results[results["fold"] == results["fold"].max()]
        .pivot(index="hotel_id", columns="stay_date", values="reconciled")
        .loc[world.hotels["hotel_id"]]
        .to_numpy()
    )
    capped, nodes = reconcile.cap_and_aggregate(pivot, capacity, summing)
    return float(np.abs(nodes[0] - capped.sum(axis=0)).max())


def drift_report(results: pd.DataFrame, world: World, window: int = 28) -> pd.DataFrame:
    """Hotels whose recent forecast error stays one-sided: candidates for a manual review."""
    last_fold = results[(results["fold"] == results["fold"].max()) & (results["horizon"] <= window)]
    grouped = last_fold.groupby("hotel_id")
    report = pd.DataFrame(
        {
            "bias_28d": grouped.apply(
                lambda g: metrics.bias(g["actual_rooms"], g["forecast_rooms"]),
                include_groups=False,
            ),
            "y1_change_28d": grouped.apply(
                lambda g: g["actual_rooms"].sum() / g["y1_rooms"].sum() - 1, include_groups=False
            ),
        }
    ).reset_index()
    flags = world.hotels[["hotel_id", "brand", "region", "is_drifting"]]
    report = report.merge(flags, on="hotel_id").sort_values("y1_change_28d")
    return report.reset_index(drop=True)


def write_reports(results: pd.DataFrame, world: World, out_dir: Path, horizons: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    accuracy_by_horizon(results, horizons).to_csv(out_dir / "accuracy_by_horizon.csv", index=False)
    for by in ("brand", "region"):
        accuracy_by_segment(results, by).to_csv(out_dir / f"accuracy_by_{by}.csv", index=False)
    drift_report(results, world).to_csv(out_dir / "drift_report.csv", index=False)
    accuracy_by_level(results).to_csv(out_dir / "accuracy_by_level.csv", index=False)
    summary = {
        "interval_80_coverage": interval_coverage(results),
        "coherence_gap_rooms": group_coherence_gap(results, world),
        "folds": int(results["fold"].nunique()),
        "hotels": int(results["hotel_id"].nunique()),
        "rows": len(results),
    }
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
