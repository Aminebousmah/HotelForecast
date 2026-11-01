"""Figures of the README, regenerated from the backtest outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter

from hotelforecast import backtest as bt
from hotelforecast.simulate import World

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = {"LightGBM global": BLUE, "RMS simulé": ORANGE, "Naïve Y-1": AQUA}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#5598e7", "#184f95"])
DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


def _axes(width: float = 8.0, height: float = 4.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def wape_by_horizon(results: pd.DataFrame, path: Path) -> None:
    fig, ax = _axes()
    columns = {"forecast_rooms": "LightGBM global", "rms": "RMS simulé", "naive_y1": "Naïve Y-1"}
    for column, label in columns.items():
        grouped = results.groupby("horizon")
        wape = grouped.apply(
            lambda g, c=column: np.abs(g["actual_rooms"] - g[c]).sum() / g["actual_rooms"].sum(),
            include_groups=False,
        )
        smooth = wape.rolling(7, min_periods=1).mean() * 100
        ax.plot(smooth.index, smooth.to_numpy(), color=SERIES[label], linewidth=2, label=label)
        ax.annotate(
            label, (smooth.index[-1], smooth.iloc[-1]), xytext=(6, 0),
            textcoords="offset points", color=INK, fontsize=9, va="center",
        )
    ax.set_xlim(1, 106)
    ax.set_xticks([1, 15, 30, 45, 60, 75, 90])
    ax.set_ylim(0, None)
    ax.set_xlabel("Horizon de prévision (jours)", color=MUTED)
    ax.set_ylabel("WAPE (%)", color=MUTED)
    ax.legend(frameon=False, loc="lower right", labelcolor=MUTED, fontsize=9)
    _save(fig, ax, "Erreur de prévision par horizon, moyenne sur 7 jours", path)


def hotel_example(results: pd.DataFrame, world: World, path: Path) -> None:
    """Realised, forecast and RMS curves for the largest hotel of the last fold."""
    hotel_id = world.hotels.sort_values("capacity")["hotel_id"].iloc[-1]
    fold = results[(results["fold"] == results["fold"].max()) & (results["hotel_id"] == hotel_id)]
    fold = fold.sort_values("stay_date")
    fig, ax = _axes(8.0, 4.0)
    ax.fill_between(
        fold["stay_date"], fold["p10_rooms"], fold["p90_rooms"],
        color=BLUE, alpha=0.14, linewidth=0, label="Intervalle à 80 %",
    )
    ax.plot(fold["stay_date"], fold["actual_rooms"], color=INK, linewidth=2, label="Réalisé")
    ax.plot(fold["stay_date"], fold["forecast_rooms"], color=BLUE, linewidth=2, label="LightGBM")
    ax.plot(fold["stay_date"], fold["rms"], color=ORANGE, linewidth=2, label="RMS simulé")
    ax.set_ylabel("Chambres vendues", color=MUTED)
    ax.legend(frameon=False, ncols=4, loc="upper left", labelcolor=MUTED, fontsize=9)
    ax.set_ylim(0, fold["capacity"].iloc[0] * 1.2)
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    title = f"Hôtel {hotel_id} : 90 jours prévus depuis le {fold['origin_date'].iloc[0]:%d/%m/%Y}"
    _save(fig, ax, title, path)


def error_heatmap(results: pd.DataFrame, path: Path) -> None:
    table = bt.error_heatmap(results) * 100
    fig, ax = plt.subplots(figsize=(7.4, 3.8), facecolor=SURFACE)
    image = ax.imshow(table.to_numpy(), cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(7), DAYS)
    ax.set_yticks(range(len(table)), [MONTHS[m - 1] for m in table.index])
    ax.tick_params(colors=MUTED, length=0)
    ax.spines[:].set_visible(False)
    threshold = np.nanmean(table.to_numpy())
    for (row, column), value in np.ndenumerate(table.to_numpy()):
        if not np.isnan(value):
            ink = "#ffffff" if value > threshold else INK
            ax.text(column, row, f"{value:.0f}", ha="center", va="center", color=ink, fontsize=8)
    fig.colorbar(image, ax=ax, label="WAPE (%)", fraction=0.03)
    _save(fig, ax, "Où se trompe-t-on ? WAPE de J+1 à J+90, par mois et jour", path)


def pickup_curves(world: World, path: Path) -> None:
    """Share of the final rooms on the books, per lead, over the last simulated year."""
    final = world.rooms_sold[:, -365:].sum()
    otb = world.otb[:, -365:, :91].sum(axis=(0, 1))
    leads = np.arange(91)
    fig, ax = _axes(8.0, 3.8)
    ax.plot(leads, 100 * otb / final, color=BLUE, linewidth=2)
    for lead in (30, 14, 7):
        share = 100 * otb[lead] / final
        ax.plot([lead], [share], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE)
        ax.annotate(f"J-{lead} : {share:.0f} %", (lead, share), xytext=(8, -14),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("Jours avant le séjour", color=MUTED)
    ax.set_ylabel("Part du final en portefeuille (%)", color=MUTED)
    _save(fig, ax, "Courbe de montée en charge du groupe", path)


def write_figures(results: pd.DataFrame, world: World, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wape_by_horizon(results, out_dir / "wape_by_horizon.png")
    hotel_example(results, world, out_dir / "hotel_example.png")
    error_heatmap(results, out_dir / "error_heatmap.png")
    pickup_curves(world, out_dir / "pickup_curve.png")
