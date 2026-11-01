"""Command line entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from hotelforecast import backtest as bt
from hotelforecast import contract as ct
from hotelforecast.simulate import SimConfig, daily_fact, load_world, save_world, simulate

app = typer.Typer(no_args_is_help=True, add_completion=False)
DATA_DIR = Path("data/synthetic")
OUTPUT_DIR = Path("data/outputs")
REPORTS_DIR = Path("reports")


@app.command("simulate")
def simulate_command(
    hotels: int = typer.Option(60, help="Nombre d'hôtels fictifs."),
    years: int = typer.Option(4, help="Années simulées."),
    seed: int = typer.Option(42, help="Graine du générateur."),
    start: str = typer.Option("2022-01-01", help="Première date de séjour."),
) -> None:
    """Génère le groupe hôtelier fictif."""
    world = simulate(SimConfig(n_hotels=hotels, years=years, seed=seed, start=start))
    save_world(world, DATA_DIR)
    typer.echo(f"{len(world.reservations):,} demandes de réservation écrites dans {DATA_DIR}")


@app.command("load")
def load_command(host: str = "localhost", port: int = 8123) -> None:
    """Charge les données brutes dans ClickHouse."""
    import clickhouse_connect

    world = load_world(DATA_DIR)
    client = clickhouse_connect.get_client(host=host, port=port)
    client.command("create database if not exists raw")
    tables = {
        "raw.daily_hotel": (
            daily_fact(world).fillna({"adr": 0.0}),
            "hotel_id String, stay_date Date, capacity UInt16, rooms_sold UInt16, "
            "rooms_occupied UInt16, rooms_denied UInt16, room_revenue Float64, occupancy Float32, "
            "adr Float32, revpar Float32",
            "(hotel_id, stay_date)",
        ),
        "raw.hotels": (
            world.hotels.drop(columns=["is_drifting"]),
            "hotel_id String, brand String, region String, capacity UInt16, "
            "business_share Float32, base_load Float32, price_index Float32",
            "hotel_id",
        ),
        "raw.dates": (
            world.calendar,
            "date Date, dow UInt8, month UInt8, doy UInt16, is_public_holiday Bool, "
            "school_holiday_share Float32",
            "date",
        ),
    }
    backtest_path = OUTPUT_DIR / "backtest.parquet"
    if backtest_path.exists():
        columns = [
            "fold", "hotel_id", "origin_date", "stay_date", "horizon", "actual_rooms",
            "forecast_rooms", "p10_rooms", "p90_rooms", "rms", "naive_y1",
        ]
        tables["raw.backtest"] = (
            pd.read_parquet(backtest_path, columns=columns),
            "fold UInt8, hotel_id String, origin_date Date, stay_date Date, horizon UInt8, "
            "actual_rooms Float32, forecast_rooms Float32, p10_rooms Float32, p90_rooms Float32, "
            "rms Float32, naive_y1 Float32",
            "(fold, hotel_id, stay_date)",
        )
    for table, (frame, schema, order) in tables.items():
        client.command(f"drop table if exists {table}")
        client.command(f"create table {table} ({schema}) engine = MergeTree order by {order}")
        client.insert_df(table, frame)
        typer.echo(f"{table}: {len(frame):,} lignes")


@app.command("backtest")
def backtest_command(
    horizons: str = typer.Option("7,30,90", help="Horizons rapportés."),
    folds: int = typer.Option(6, help="Nombre d'origines glissantes."),
) -> None:
    """Lance l'évaluation à origine glissante et régénère les rapports."""
    world = load_world(DATA_DIR)
    results = bt.run_backtest(world, n_folds=folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_DIR / "backtest.parquet", index=False)
    bt.write_reports(results, world, REPORTS_DIR, [int(h) for h in horizons.split(",")])
    typer.echo(f"Couverture de l'intervalle à 80 % : {bt.interval_coverage(results):.1%}")
    typer.echo(f"Écart de cohérence groupe/hôtels : {bt.group_coherence_gap(results, world):.6f}")
    typer.echo((REPORTS_DIR / "accuracy_by_horizon.csv").read_text(encoding="utf-8"))


@app.command("report")
def report_command() -> None:
    """Régénère les rapports et les figures du README à partir du dernier backtest."""
    from hotelforecast.report import write_figures

    world = load_world(DATA_DIR)
    results = pd.read_parquet(OUTPUT_DIR / "backtest.parquet")
    bt.write_reports(results, world, REPORTS_DIR, [7, 30, 90])
    write_figures(results, world, REPORTS_DIR / "figures")
    typer.echo(f"Rapports écrits dans {REPORTS_DIR}")


@app.command("forecast")
def forecast_command(
    asof: str = typer.Option("2025-06-01", help="Date de prévision."),
    out: Path = typer.Option(OUTPUT_DIR / "demand_forecast.parquet", help="Fichier de sortie."),
) -> None:
    """Produit le contrat de prévision J+1 à J+90 consommé par NoShow."""
    world = load_world(DATA_DIR)
    dates = pd.DatetimeIndex(world.calendar["date"])
    origin = int(dates.get_loc(pd.Timestamp(asof)))
    contract = ct.to_contract(ct.forecast_from(world, origin))
    ct.write_contract(contract, out)
    typer.echo(f"{len(contract):,} lignes écrites dans {out}")


@app.command("rolling-contract")
def rolling_contract_command(
    start: str = typer.Option("2025-07-01", help="Première nuit prévue."),
    end: str = typer.Option("2025-12-31", help="Dernière nuit prévue."),
    horizon: int = typer.Option(7, help="Délai entre la prévision et la nuit."),
    out: Path = typer.Option(OUTPUT_DIR / "demand_forecast_rolling.parquet", help="Sortie."),
) -> None:
    """Rejoue une période de test à J-x, pour le backtest des décisions de NoShow."""
    world = load_world(DATA_DIR)
    dates = pd.DatetimeIndex(world.calendar["date"])
    first, last = int(dates.get_loc(pd.Timestamp(start))), int(dates.get_loc(pd.Timestamp(end)))
    contract = ct.to_contract(ct.rolling_forecast(world, first, last, horizon))
    ct.write_contract(contract, out)
    typer.echo(f"{len(contract):,} lignes écrites dans {out}")


@app.command("export-bookings")
def export_bookings_command(
    out: Path = typer.Option(OUTPUT_DIR / "bookings.parquet", help="Fichier de sortie."),
    hotels: int = typer.Option(12, help="Nombre d'hôtels exportés."),
) -> None:
    """Exporte le journal de réservations d'un sous-ensemble d'hôtels pour NoShow."""
    world = load_world(DATA_DIR)
    selected = world.hotels["hotel_id"].head(hotels)
    bookings = world.reservations[world.reservations["hotel_id"].isin(selected)]
    out.parent.mkdir(parents=True, exist_ok=True)
    bookings.to_parquet(out, index=False, compression="zstd")
    world.hotels[world.hotels["hotel_id"].isin(selected)].drop(columns=["is_drifting"]).to_parquet(
        out.with_name("hotels.parquet"), index=False
    )
    typer.echo(f"{len(bookings):,} réservations écrites dans {out}")


if __name__ == "__main__":
    app()
