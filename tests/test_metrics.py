import numpy as np
import pytest

from hotelforecast import metrics


def test_wape_weights_by_volume():
    assert metrics.wape([100, 0], [90, 10]) == pytest.approx(0.2)


def test_wape_all_zero_actuals_is_nan():
    assert np.isnan(metrics.wape([0, 0], [1, 1]))


def test_mape_skips_zero_actuals():
    assert metrics.mape([0, 100], [5, 90]) == pytest.approx(0.1)


def test_smape_is_symmetric_and_bounded():
    assert metrics.smape([100], [50]) == pytest.approx(metrics.smape([50], [100]))
    assert metrics.smape([1], [0]) == pytest.approx(2.0)


def test_bias_sign():
    assert metrics.bias([100], [110]) == pytest.approx(0.1)
    assert metrics.bias([100], [90]) == pytest.approx(-0.1)


def test_coverage_includes_bounds():
    assert metrics.coverage([1, 2, 3, 4], [1, 1, 1, 1], [3, 3, 3, 3]) == pytest.approx(0.75)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.wape([1, 2], [1])
