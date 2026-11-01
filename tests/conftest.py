import pytest

from hotelforecast.simulate import SimConfig, World, simulate


@pytest.fixture(scope="session")
def world() -> World:
    return simulate(SimConfig(n_hotels=6, years=3, seed=7))
