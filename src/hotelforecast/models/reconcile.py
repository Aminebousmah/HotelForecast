"""Hierarchical reconciliation (hotel -> brand/region -> brand -> group).

Reconciliation can push a hotel above its capacity, so caps are applied afterwards at the
bottom level and every aggregate is rebuilt from the hotels. Coherence then holds exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summing_matrix(hotels: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return S (n_nodes x n_hotels) and node names, from the group down to the hotels."""
    columns: list[np.ndarray] = [np.ones(len(hotels))]
    names = ["group"]
    for brand in sorted(hotels["brand"].unique()):
        columns.append((hotels["brand"] == brand).to_numpy(dtype=float))
        names.append(f"brand={brand}")
    pairs = hotels[["brand", "region"]].drop_duplicates().sort_values(["brand", "region"])
    for brand, region in pairs.itertuples(index=False):
        mask = (hotels["brand"] == brand) & (hotels["region"] == region)
        columns.append(mask.to_numpy(dtype=float))
        names.append(f"brand={brand}/region={region}")
    return np.vstack([*columns, np.eye(len(hotels))]), names + hotels["hotel_id"].tolist()


def wls_struct(base: np.ndarray, summing: np.ndarray) -> np.ndarray:
    """MinT with structural weights. `base` is (n_nodes, n_steps); returns hotel-level forecasts."""
    weights = summing.sum(axis=1)
    weighted = summing.T / weights
    return np.linalg.solve(weighted @ summing, weighted @ base)


def cap_and_aggregate(
    bottom: np.ndarray, capacity: np.ndarray, summing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clip hotels to [0, capacity], then recompute all nodes so that totals stay coherent."""
    capped = np.clip(bottom, 0, capacity[:, None])
    return capped, summing @ capped
