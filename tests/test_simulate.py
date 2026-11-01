import numpy as np
import pandas as pd

from hotelforecast.simulate import SimConfig, daily_fact, simulate


def test_generator_is_deterministic():
    first = simulate(SimConfig(n_hotels=2, years=1, seed=11))
    second = simulate(SimConfig(n_hotels=2, years=1, seed=11))
    assert np.array_equal(first.rooms_sold, second.rooms_sold)
    pd.testing.assert_frame_equal(first.reservations, second.reservations)


def test_seed_changes_the_world():
    first = simulate(SimConfig(n_hotels=2, years=1, seed=1))
    second = simulate(SimConfig(n_hotels=2, years=1, seed=2))
    assert not np.array_equal(first.rooms_sold, second.rooms_sold)


def test_sales_never_exceed_capacity(world):
    capacity = world.hotels["capacity"].to_numpy()[:, None]
    assert (world.rooms_sold <= capacity).all()
    assert (world.otb.max(axis=2) <= capacity).all()


def test_otb_at_arrival_equals_rooms_sold(world):
    assert np.array_equal(world.otb[:, :, 0], world.rooms_sold)


def test_occupied_never_exceeds_sold(world):
    assert (world.rooms_occupied <= world.rooms_sold).all()


def test_revpar_is_occupancy_times_adr(world):
    fact = daily_fact(world).dropna(subset=["adr"])
    product = fact["occupancy"] * fact["adr"]
    assert np.allclose(product, fact["revpar"], atol=0.05)


def test_denied_requests_are_kept_for_counterfactuals(world):
    assert (~world.reservations["accepted"]).any()
    assert world.rooms_denied.sum() > 0
