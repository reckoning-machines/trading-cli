#!/usr/bin/env python3
# Banks Pair-Trading Controller CLI

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.config import get_config_or_exit
from src.storage import ensure_directories
from src.universe import load_universe
from src.tickets import load_active_tickets, get_required_pairs
from src.fmp_client import FMPClient, compute_data_checksum
from src.storage import write_daily_bars, write_pull_log, read_daily_bars
from src.features_pair_state import compute_pair_features, build_pair_state_dataframe
from src.storage import write_pair_state_features
from src.run_engine import run_controller, validate_run_artifacts
from src.reporting import (
    print_ingest_summary,
    print_features_summary,
    print_controller_summary,
    generate_run_report,
)

app = typer.Typer(help="Banks Pair-Trading Controller CLI")
console = Console()


@app.command("ingest-prices")
def ingest_prices(
    start: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
):
    """Ingest daily price data from FMP for all universe symbols."""
    config = get_config_or_exit()
    ensure_directories(config)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    universe = load_universe(config)
    symbols = universe.all_symbols

    console.print(f"\n[bold]Ingesting prices for {len(symbols)} symbols[/bold]")
    console.print(f"Date range: {start_date} to {end_date}\n")

    client = FMPClient(config)
    results = []

    for symbol in symbols:
        try:
            console.print(f"Fetching {symbol}...", end=" ")
            df = client.fetch_daily_bars(symbol, start_date, end_date)

            if df.is_empty():
                console.print("[yellow]no data[/yellow]")
                results.append({
                    "symbol": symbol,
                    "rows": 0,
                    "last_date": None,
                    "status": "no data",
                })
                continue

            write_daily_bars(config, symbol, df)

            checksum = compute_data_checksum(df)
            write_pull_log(
                config=config,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                row_count=len(df),
                data_checksum=checksum,
            )

            last_date = df.select("date").max().item()
            console.print(f"[green]{len(df)} rows[/green]")

            results.append({
                "symbol": symbol,
                "rows": len(df),
                "last_date": str(last_date) if last_date else None,
                "status": "ok",
            })

        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
            results.append({
                "symbol": symbol,
                "rows": 0,
                "last_date": None,
                "status": f"error: {e}",
            })

    console.print()
    print_ingest_summary(results, console)


@app.command("build-features")
def build_features(
    asof: str = typer.Option(..., help="As-of date (YYYY-MM-DD)"),
    feature_version: str = typer.Option("v1", help="Feature version string"),
    window_hedge: int = typer.Option(60, help="Rolling window for beta calculation"),
    window_z: int = typer.Option(60, help="Rolling window for z-score calculation"),
):
    """Build pair state features for active tickets."""
    config = get_config_or_exit()
    ensure_directories(config)

    asof_date = date.fromisoformat(asof)

    tickets = load_active_tickets(config, asof_date)
    if not tickets:
        console.print("[yellow]No active tickets found[/yellow]")
        return

    pairs = get_required_pairs(tickets)
    console.print(f"\n[bold]Building features for {len(pairs)} pairs[/bold]")
    console.print(f"As-of date: {asof_date}")
    console.print(f"Feature version: {feature_version}")
    console.print(f"Window hedge: {window_hedge}, Window z: {window_z}\n")

    features_list = []

    for leg_a, leg_b in pairs:
        console.print(f"Computing {leg_a}__{leg_b}...", end=" ")

        features = compute_pair_features(
            config=config,
            leg_a=leg_a,
            leg_b=leg_b,
            asof_date=asof_date,
            window_hedge=window_hedge,
            window_z=window_z,
            feature_version=feature_version,
        )

        if features is None:
            console.print("[yellow]insufficient data[/yellow]")
            continue

        features_list.append(features)
        console.print(f"[green]beta={features['beta']:.4f}, z={features['zscore']:.4f}[/green]")

    if not features_list:
        console.print("\n[red]No features computed - check price data availability[/red]")
        return

    df = build_pair_state_dataframe(features_list)
    write_pair_state_features(config, feature_version, asof_date, df)

    console.print()
    print_features_summary(features_list, console)
    console.print(f"\n[green]Features written to pair_state partition[/green]")


@app.command("run-controller")
def run_controller_cmd(
    asof: str = typer.Option(..., help="As-of date (YYYY-MM-DD)"),
    feature_version: str = typer.Option("v1", help="Feature version string"),
    controller: str = typer.Option("baseline", help="Controller name (baseline or rl)"),
    seed: int = typer.Option(123, help="Random seed for determinism"),
):
    """Run the trading controller to generate position recommendations."""
    config = get_config_or_exit()
    ensure_directories(config)

    asof_date = date.fromisoformat(asof)

    console.print(f"\n[bold]Running controller[/bold]")
    console.print(f"As-of date: {asof_date}")
    console.print(f"Feature version: {feature_version}")
    console.print(f"Controller: {controller}")
    console.print(f"Seed: {seed}\n")

    try:
        result = run_controller(
            config=config,
            asof_date=asof_date,
            feature_version=feature_version,
            controller_name=controller,
            seed=seed,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    run_id = result["run_id"]
    run_dir = result["run_dir"]
    decisions = result["decisions"]
    metrics = result["metrics"]

    console.print(f"[green]Run completed: {run_id}[/green]\n")

    validation_errors = validate_run_artifacts(run_dir)
    if validation_errors:
        console.print("[yellow]Validation warnings:[/yellow]")
        for err in validation_errors:
            console.print(f"  - {err}")
        console.print()

    print_controller_summary(decisions, console)

    console.print(f"\n[bold]Metrics:[/bold]")
    console.print(f"  Total tickets: {metrics['total_tickets']}")
    console.print(f"  Tickets with position: {metrics['tickets_with_position']}")
    console.print(f"  Tickets clamped: {metrics['tickets_clamped']}")
    console.print(f"  Total gross notional: ${metrics['total_gross_notional']:,.0f}")
    console.print(f"  Est. costs: ${metrics['total_est_costs']:,.2f}")
    console.print(f"  Est. daily vol: ${metrics['total_est_daily_vol']:,.0f}")

    console.print(f"\n[bold]Run directory:[/bold] {run_dir}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to report on"),
):
    """Generate a report for a completed run."""
    config = get_config_or_exit()

    run_dir = config.runs_dir / f"run_id={run_id}"

    if not run_dir.exists():
        for d in config.runs_dir.iterdir():
            if d.is_dir() and run_id in d.name:
                run_dir = d
                break

    if not run_dir.exists():
        console.print(f"[red]Run directory not found: {run_id}[/red]")
        raise typer.Exit(1)

    generate_run_report(run_dir, console)


@app.command("list-runs")
def list_runs():
    """List all completed runs."""
    config = get_config_or_exit()

    if not config.runs_dir.exists():
        console.print("[yellow]No runs directory found[/yellow]")
        return

    runs = sorted(config.runs_dir.iterdir(), reverse=True)

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    console.print("\n[bold]Available runs:[/bold]\n")
    for run_dir in runs[:20]:
        if run_dir.is_dir() and run_dir.name.startswith("run_id="):
            run_id = run_dir.name.replace("run_id=", "")
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                import json
                with open(manifest_path) as f:
                    manifest = json.load(f)
                asof = manifest.get("asof_date", "?")
                ctrl = manifest.get("controller_version", "?")
                console.print(f"  {run_id}  [dim](asof={asof}, controller={ctrl})[/dim]")
            else:
                console.print(f"  {run_id}")


@app.command("list-tickets")
def list_tickets():
    """List all active tickets."""
    config = get_config_or_exit()

    tickets = load_active_tickets(config, date.today())

    if not tickets:
        console.print("[yellow]No active tickets found[/yellow]")
        return

    console.print("\n[bold]Active tickets:[/bold]\n")
    for ticket in tickets:
        console.print(f"  {ticket.ticket_id}")
        console.print(f"    Pair: {ticket.pair_id}")
        console.print(f"    Orientation: {ticket.orientation}")
        console.print(f"    Max gross: ${ticket.max_gross_notional:,.0f}")
        console.print(f"    Conviction: {ticket.conviction}")
        console.print(f"    Expires: {ticket.expires_on}")
        console.print()


@app.command("test-harness")
def test_harness():
    """Run the test harness to validate system setup."""
    config = get_config_or_exit()

    from src.run_engine import run_test_harness

    console.print("\n[bold]Running test harness[/bold]\n")
    success = run_test_harness(config)

    if success:
        console.print("\n[green]All tests passed[/green]")
    else:
        console.print("\n[red]Some tests failed[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
