import copy

import numpy as np
import pandas as pd

from hotelforecast.features import FEATURE_COLUMNS, FeatureBuilder, aligned_lag, pickup_estimate


def test_aligned_lag_shifts_and_pads():
    result = aligned_lag(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert np.isnan(result[:2]).all()
    assert result[2:].tolist() == [1.0, 2.0]


def test_aligned_lag_keeps_day_of_week():
    dates = pd.date_range("2024-01-01", periods=800)
    assert (dates[364:].day_of_week == dates[:-364].day_of_week).all()


def test_pickup_only_uses_stays_realised_at_origin():
    otb = np.zeros((1, 200, 121))
    sold = np.full((1, 200), 10.0)
    sold[0, 100:] = 999.0  # future stays must not influence the estimate
    assert pickup_estimate(otb, sold, stay=110, horizon=10, origin=100)[0] == 10.0


def test_no_future_information_leaks_into_features(world):
    """Rewriting everything after the origin must leave the features unchanged."""
    origin, horizons = 800, np.arange(1, 91)
    reference = FeatureBuilder(world).rows(origin, horizons)

    altered = copy.deepcopy(world)
    n_days = altered.rooms_sold.shape[1]
    altered.rooms_sold[:, origin + 1 :] = 1
    altered.room_revenue[:, origin + 1 :] = 1.0
    stays, leads = np.meshgrid(np.arange(n_days), np.arange(121), indexing="ij")
    altered.otb[:, stays - leads > origin] = 1
    perturbed = FeatureBuilder(altered).rows(origin, horizons)

    pd.testing.assert_frame_equal(reference[FEATURE_COLUMNS], perturbed[FEATURE_COLUMNS])
