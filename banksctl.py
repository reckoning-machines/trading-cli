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
    print_realized_summary,
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


@app.command("train-policy")
def train_policy_cmd(
    train_start: str = typer.Option(..., help="Training start date (YYYY-MM-DD)"),
    train_end: str = typer.Option(..., help="Training end date (YYYY-MM-DD)"),
    eval_start: str = typer.Option(..., help="Evaluation start date (YYYY-MM-DD)"),
    eval_end: str = typer.Option(..., help="Evaluation end date (YYYY-MM-DD)"),
    feature_version: str = typer.Option("v1", help="Feature version string"),
    label_version: str = typer.Option("v1", help="Label version string"),
    model_version: str = typer.Option(..., help="Model version string to save"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    lambda_reg: float = typer.Option(1.0, help="Ridge regression regularization"),
    reward_horizon: int = typer.Option(5, help="Reward horizon in trading days"),
):
    """Train RL policy using contextual bandit with ridge regression."""
    config = get_config_or_exit()
    ensure_directories(config)

    from src.train import train_policy, list_available_models

    train_start_date = date.fromisoformat(train_start)
    train_end_date = date.fromisoformat(train_end)
    eval_start_date = date.fromisoformat(eval_start)
    eval_end_date = date.fromisoformat(eval_end)

    console.print(f"\n[bold]Training RL Policy[/bold]")
    console.print(f"Training period: {train_start_date} to {train_end_date}")
    console.print(f"Evaluation period: {eval_start_date} to {eval_end_date}")
    console.print(f"Feature version: {feature_version}")
    console.print(f"Label version: {label_version}")
    console.print(f"Model version: {model_version}")
    console.print(f"Seed: {seed}")
    console.print(f"Lambda (regularization): {lambda_reg}")
    console.print(f"Reward horizon: {reward_horizon} days\n")

    try:
        report = train_policy(
            config=config,
            train_start=train_start_date,
            train_end=train_end_date,
            eval_start=eval_start_date,
            eval_end=eval_end_date,
            feature_version=feature_version,
            label_version=label_version,
            model_version=model_version,
            seed=seed,
            lambda_reg=lambda_reg,
            reward_horizon=reward_horizon,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if "error" in report:
        console.print(f"[red]Training failed: {report['error']}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Training completed successfully[/green]\n")

    console.print(f"[bold]Training Summary:[/bold]")
    console.print(f"  Training examples: {report['n_train_examples']}")
    console.print(f"  Evaluation examples: {report['n_eval_examples']}")

    train_stats = report.get("train_stats", {})
    if "action_stats" in train_stats:
        console.print(f"\n[bold]Per-Action Training Stats:[/bold]")
        for action, stats in sorted(train_stats["action_stats"].items()):
            console.print(f"  Action {action}:")
            console.print(f"    Samples: {stats.get('n_samples', 0)}")
            console.print(f"    Mean reward: ${stats.get('mean_reward', 0):.2f}")
            if "mse" in stats:
                console.print(f"    MSE: {stats['mse']:.4f}")

    eval_stats = report.get("eval_stats", {})
    if eval_stats and "error" not in eval_stats:
        console.print(f"\n[bold]Evaluation Results:[/bold]")
        console.print(f"  Baseline mean reward: ${eval_stats.get('baseline_mean_reward', 0):.2f}")
        console.print(f"  Predicted mean reward: ${eval_stats.get('predicted_mean_reward', 0):.2f}")
        console.print(f"  Agreement rate: {eval_stats.get('agreement_rate', 0)*100:.1f}%")

        console.print(f"\n[bold]Action Distribution:[/bold]")
        baseline_counts = eval_stats.get("baseline_action_counts", {})
        learned_counts = eval_stats.get("learned_action_counts", {})
        console.print(f"  {'Action':<10} {'Baseline':<10} {'Learned':<10}")
        for action in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            b_count = baseline_counts.get(action, 0)
            l_count = learned_counts.get(action, 0)
            console.print(f"  {action:<10} {b_count:<10} {l_count:<10}")

    model_dir = config.pair_policy_dir / f"version={model_version}"
    console.print(f"\n[green]Model saved to: {model_dir}[/green]")


@app.command("list-models")
def list_models():
    """List all trained RL models."""
    config = get_config_or_exit()

    from src.train import list_available_models

    models = list_available_models(config)

    if not models:
        console.print("[yellow]No trained models found[/yellow]")
        return

    console.print("\n[bold]Available Models:[/bold]\n")
    for version, info in sorted(models.items()):
        console.print(f"  [cyan]{version}[/cyan]")
        if "trained_at" in info:
            console.print(f"    Trained: {info['trained_at']}")
        if "n_train_examples" in info:
            console.print(f"    Training examples: {info['n_train_examples']}")
        if "train_start" in info and "train_end" in info:
            console.print(f"    Train period: {info['train_start']} to {info['train_end']}")
        eval_stats = info.get("eval_stats", {})
        if eval_stats and "baseline_mean_reward" in eval_stats:
            console.print(f"    Baseline reward: ${eval_stats['baseline_mean_reward']:.2f}")
            console.print(f"    Predicted reward: ${eval_stats.get('predicted_mean_reward', 0):.2f}")
        console.print()


@app.command("realize")
def realize(
    asof: str = typer.Option(..., help="As-of date (YYYY-MM-DD)"),
    run_id: str = typer.Option(..., help="Run ID to compute realized PnL for"),
    cost_bps: float = typer.Option(1.0, help="Cost in basis points per turnover"),
    label_version: str = typer.Option("v1", help="Label version string"),
):
    """Compute realized PnL labels for a completed run."""
    config = get_config_or_exit()
    ensure_directories(config)

    from src.realized_pnl import compute_realized_pnl

    asof_date = date.fromisoformat(asof)

    console.print(f"\n[bold]Computing realized PnL[/bold]")
    console.print(f"Run ID: {run_id}")
    console.print(f"As-of date: {asof_date}")
    console.print(f"Cost BPS: {cost_bps}")
    console.print(f"Label version: {label_version}\n")

    try:
        result = compute_realized_pnl(
            config=config,
            run_id=run_id,
            cost_bps=cost_bps,
            label_version=label_version,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    realized_records = result["realized_records"]
    summary = result["summary"]
    validation_errors = result["validation_errors"]
    run_dir = result["run_dir"]

    if validation_errors:
        console.print("[yellow]Validation warnings:[/yellow]")
        for err in validation_errors:
            console.print(f"  - {err}")
        console.print()

    print_realized_summary(realized_records, console)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Count OK: {summary['count_ok']}")
    console.print(f"  Count NO_PRICE_SKIP: {summary['count_no_price_skip']}")
    console.print(f"  Total PnL Gross: ${summary['total_pnl_gross']:,.2f}")
    console.print(f"  Total Costs: ${summary['total_costs']:,.2f}")
    console.print(f"  Total PnL Net: ${summary['total_pnl_net']:,.2f}")
    console.print(f"  Total Trade Notional: ${summary['total_trade_notional']:,.0f}")

    console.print(f"\n[green]Realized PnL written to:[/green]")
    console.print(f"  - {run_dir}/realized.parquet")
    console.print(f"  - data/labels/realized/version={label_version}/asof_date={asof_date}/")


if __name__ == "__main__":
    app()
