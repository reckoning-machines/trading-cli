# Reporting module - generates run reports and summaries

from pathlib import Path
from typing import Optional, Dict, Any

import polars as pl
from rich.console import Console
from rich.table import Table

from src.storage import read_decisions, read_metrics


def print_decisions_summary(run_dir: Path, console: Optional[Console] = None) -> None:
    if console is None:
        console = Console()

    decisions = read_decisions(run_dir)

    if decisions is None or decisions.is_empty():
        console.print("[yellow]No decisions found in run directory[/yellow]")
        return

    table = Table(title="Decisions Summary")
    table.add_column("Ticket ID", style="cyan")
    table.add_column("Pair ID", style="green")
    table.add_column("Action Units", justify="right")
    table.add_column("Executed Units", justify="right")
    table.add_column("Notional A", justify="right")
    table.add_column("Notional B", justify="right")
    table.add_column("Clamp Codes", style="yellow")

    for row in decisions.iter_rows(named=True):
        table.add_row(
            row["ticket_id"],
            row["pair_id"],
            f"{row['action_units']:.2f}",
            f"{row['executed_units']:.2f}",
            f"${row['executed_notional_a']:,.0f}",
            f"${row['executed_notional_b']:,.0f}",
            row["clamp_codes"] or "-",
        )

    console.print(table)


def print_metrics_summary(run_dir: Path, console: Optional[Console] = None) -> None:
    if console is None:
        console = Console()

    metrics = read_metrics(run_dir)

    if metrics is None:
        console.print("[yellow]No metrics found in run directory[/yellow]")
        return

    table = Table(title="Run Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:,.2f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def generate_run_report(run_dir: Path, console: Optional[Console] = None) -> None:
    if console is None:
        console = Console()

    run_id = run_dir.name.replace("run_id=", "")
    console.print(f"\n[bold]Run Report: {run_id}[/bold]\n")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        info_table = Table(title="Run Information", show_header=False)
        info_table.add_column("Field", style="cyan")
        info_table.add_column("Value")

        info_table.add_row("Run ID", manifest.get("run_id", "N/A"))
        info_table.add_row("As-of Date", manifest.get("asof_date", "N/A"))
        info_table.add_row("Seed", str(manifest.get("seed", "N/A")))
        info_table.add_row("Feature Version", manifest.get("feature_version", "N/A"))
        info_table.add_row("Controller Version", manifest.get("controller_version", "N/A"))
        info_table.add_row("Tickets Used", str(len(manifest.get("tickets_snapshot_paths", []))))

        console.print(info_table)
        console.print()

    print_decisions_summary(run_dir, console)
    console.print()
    print_metrics_summary(run_dir, console)


def print_ingest_summary(
    results: list,
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()

    table = Table(title="Price Ingestion Summary")
    table.add_column("Symbol", style="cyan")
    table.add_column("Rows Pulled", justify="right")
    table.add_column("Last Date", style="green")
    table.add_column("Status", style="yellow")

    for result in results:
        table.add_row(
            result["symbol"],
            str(result["rows"]),
            result["last_date"] or "N/A",
            result["status"],
        )

    console.print(table)


def print_features_summary(
    features: list,
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()

    table = Table(title="Pair State Features")
    table.add_column("Pair ID", style="cyan")
    table.add_column("Beta", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("Extreme Z", justify="center")
    table.add_column("Beta Stability", justify="right")

    for feat in features:
        extreme_display = "[red]YES[/red]" if feat["extreme_z"] == 1 else "[green]NO[/green]"
        table.add_row(
            feat["pair_id"],
            f"{feat['beta']:.4f}",
            f"{feat['zscore']:.4f}",
            extreme_display,
            f"{feat['beta_stability']:.4f}",
        )

    console.print(table)


def print_controller_summary(
    decisions: list,
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()

    table = Table(title="Controller Decisions")
    table.add_column("Ticket ID", style="cyan")
    table.add_column("Pair ID", style="green")
    table.add_column("Z-Score", justify="right")
    table.add_column("Action", justify="right")
    table.add_column("Executed", justify="right")
    table.add_column("Notional A", justify="right")
    table.add_column("Notional B", justify="right")
    table.add_column("Clamps", style="yellow")

    for dec in decisions:
        table.add_row(
            dec["ticket_id"],
            dec["pair_id"],
            f"{dec.get('zscore', 0):.2f}",
            f"{dec['action_units']:.2f}",
            f"{dec['executed_units']:.2f}",
            f"${dec['executed_notional_a']:,.0f}",
            f"${dec['executed_notional_b']:,.0f}",
            dec["clamp_codes"] or "-",
        )

    console.print(table)


def print_realized_summary(
    records: list,
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()

    table = Table(title="Realized PnL")
    table.add_column("Ticket ID", style="cyan")
    table.add_column("Pair ID", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Ret A", justify="right")
    table.add_column("Ret B", justify="right")
    table.add_column("PnL Gross", justify="right")
    table.add_column("Costs", justify="right")
    table.add_column("PnL Net", justify="right")

    for rec in records:
        status_display = "[green]OK[/green]" if rec["status"] == "OK" else "[yellow]SKIP[/yellow]"

        pnl_color = "green" if rec["pnl_net"] >= 0 else "red"

        table.add_row(
            rec["ticket_id"],
            rec["pair_id"],
            status_display,
            f"{rec['ret_a']*100:.2f}%",
            f"{rec['ret_b']*100:.2f}%",
            f"${rec['pnl_gross']:,.2f}",
            f"${rec['costs']:,.2f}",
            f"[{pnl_color}]${rec['pnl_net']:,.2f}[/{pnl_color}]",
        )

    console.print(table)
