import numpy as np
import pandas as pd

from hotelforecast import contract as ct
from hotelforecast.backtest import fold_origins
from hotelforecast.features import FeatureBuilder
from hotelforecast.models import baseline, lgbm, reconcile

HOTELS = pd.DataFrame(
    {
        "hotel_id": ["A", "B", "C"],
        "brand": ["economy", "economy", "upscale"],
        "region": ["idf", "paca", "idf"],
        "capacity": [10, 20, 30],
    }
)


def test_summing_matrix_rebuilds_the_group_total():
    summing, names = reconcile.summing_matrix(HOTELS)
    assert names[0] == "group"
    assert (summing @ np.array([1.0, 2.0, 3.0]))[0] == 6.0
    assert names[-3:] == ["A", "B", "C"]


def test_reconciliation_leaves_coherent_forecasts_untouched():
    summing, _ = reconcile.summing_matrix(HOTELS)
    bottom = np.array([[5.0], [7.0], [11.0]])
    assert np.allclose(reconcile.wls_struct(summing @ bottom, summing), bottom)


def test_capping_keeps_totals_coherent():
    summing, _ = reconcile.summing_matrix(HOTELS)
    bottom = np.array([[15.0], [-3.0], [11.0]])
    capped, nodes = reconcile.cap_and_aggregate(bottom, HOTELS["capacity"].to_numpy(), summing)
    assert capped.ravel().tolist() == [10.0, 0.0, 11.0]
    assert nodes[0, 0] == capped.sum()


def test_folds_stay_inside_the_data_and_precede_each_other():
    origins = fold_origins(1461, 6)
    assert origins == sorted(origins)
    assert origins[-1] + 90 == 1460


def test_naive_uses_aligned_weekday(world):
    rows = FeatureBuilder(world).rows(800, np.array([10]))
    expected = world.rooms_sold[:, 810 - 364].astype(float)
    assert np.array_equal(baseline.naive_y1(world, rows), expected)


def test_forecasts_respect_capacity_and_quantile_order(world):
    builder = FeatureBuilder(world)
    model = lgbm.DemandModel()
    original = lgbm.PARAMS["n_estimators"]
    lgbm.PARAMS["n_estimators"] = 15
    try:
        model.fit(lgbm.training_rows(world, builder, 800, step=30))
    finally:
        lgbm.PARAMS["n_estimators"] = original
    rows = builder.rows(800, np.arange(1, 91))
    predicted = model.predict(rows)
    capacity = rows["capacity"].to_numpy()
    assert (predicted["p90_rooms"].to_numpy() <= capacity).all()
    assert (predicted["p10_rooms"].to_numpy() >= 0).all()
    assert (predicted["p10_rooms"] <= predicted["forecast_rooms"]).all()
    assert (predicted["forecast_rooms"] <= predicted["p90_rooms"]).all()

    frame = pd.concat([rows, predicted], axis=1)
    frame["hotel_id"] = world.hotels["hotel_id"].to_numpy()[frame["hotel_idx"].to_numpy()]
    frame["asof_date"] = world.calendar["date"].iloc[800]
    frame["stay_date"] = world.calendar["date"].to_numpy()[frame["stay_idx"].to_numpy()]
    contract = ct.to_contract(frame)
    assert list(contract.columns) == ct.CONTRACT_COLUMNS
    assert contract["forecast_unmet_rooms"].min() >= 0
