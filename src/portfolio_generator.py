# Portfolio Generator - generates ideal portfolio with exact net zero exposure

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import polars as pl

from src.config import Config, CASH_SYMBOL, NET_MODE_EXACT_ZERO
from src.storage import (
    ensure_directories,
    create_run_directory,
    write_manifest,
    write_logs,
    read_pair_state_features,
    get_fmp_cache_summary,
)
from src.tickets import PairTicket, load_active_tickets, snapshot_tickets
from src.risk_governor import apply_risk_governor
from src import controller_baseline
from src import controller_rl_stub
from src.run_engine import get_git_sha


# Portfolio-level clamp codes
CLAMP_NAME_GROSS = "NAME_GROSS_CLAMP"
CLAMP_PORTFOLIO_GROSS = "PORTFOLIO_GROSS_CLAMP"
CLAMP_NET_TO_CASH = "NET_TO_CASH"


def generate_portfolio_run_id(asof_date: date, seed: int) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    hash_input = f"portfolio_{ts}_{seed}".encode()
    short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    return f"portfolio_{ts}_{short_hash}"


def generate_ideal_portfolio(
    config: Config,
    asof_date: date,
    feature_version: str,
    controller_name: str,
    seed: int,
    portfolio_max_gross_notional: float,
    portfolio_max_name_gross_notional: float,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    # Validate portfolio constraints
    if portfolio_max_gross_notional <= 0:
        raise ValueError("portfolio_max_gross_notional must be > 0")
    if portfolio_max_name_gross_notional <= 0:
        raise ValueError("portfolio_max_name_gross_notional must be > 0")

    ensure_directories(config)
    np.random.seed(seed)

    run_id = generate_portfolio_run_id(asof_date, seed)
    run_dir = create_run_directory(config, run_id)

    logs = []
    logs.append(f"Portfolio run started: {run_id}")
    logs.append(f"As-of date: {asof_date}")
    logs.append(f"Feature version: {feature_version}")
    logs.append(f"Controller: {controller_name}")
    logs.append(f"Seed: {seed}")
    logs.append(f"Portfolio max gross: {portfolio_max_gross_notional:,.0f}")
    logs.append(f"Portfolio max name gross: {portfolio_max_name_gross_notional:,.0f}")
    logs.append(f"Net mode: {NET_MODE_EXACT_ZERO}")

    # Load active tickets
    tickets = load_active_tickets(config, asof_date)
    logs.append(f"Loaded {len(tickets)} active tickets")

    if not tickets:
        logs.append("No active tickets - empty portfolio")
        write_logs(run_dir, "\n".join(logs))
        return _create_empty_portfolio_result(config, run_dir, run_id, asof_date, feature_version, logs)

    # Snapshot tickets
    snapshot_result = snapshot_tickets(tickets, run_dir)

    # Load features
    features_df = read_pair_state_features(config, feature_version, asof_date)
    if features_df is None:
        logs.append(f"ERROR: No features found for {asof_date} with version {feature_version}")
        logs.append("Run: banksctl build-features --asof {asof_date}")
        write_logs(run_dir, "\n".join(logs))
        raise ValueError(
            f"No features found for {asof_date} with version {feature_version}. "
            f"Run: banksctl build-features --asof {asof_date}"
        )

    # Select controller
    if controller_name == "baseline":
        controller_module = controller_baseline
    elif controller_name == "rl":
        controller_module = controller_rl_stub
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    controller_version = controller_module.get_controller_version()
    if controller_name == "rl" and model_version:
        controller_version = f"rl_bandit_{model_version}"

    # Step 1-3: Compute per-ticket decisions
    ticket_decisions = []

    for ticket in tickets:
        pair_id = ticket.pair_id

        pair_features = features_df.filter(pl.col("pair_id") == pair_id)
        if pair_features.is_empty():
            logs.append(f"WARNING: No features for pair {pair_id}, skipping ticket {ticket.ticket_id}")
            continue

        if pair_features.height > 1:
            raise ValueError(f"Multiple feature rows for pair {pair_id}")

        feat_row = pair_features.row(0, named=True)
        beta = feat_row["beta"]
        zscore = feat_row["zscore"]
        extreme_z = feat_row["extreme_z"]

        if not np.isfinite(beta) or not np.isfinite(zscore):
            logs.append(f"WARNING: Non-finite beta or zscore for {pair_id}, skipping")
            continue

        # Compute action using controller
        if controller_name == "rl" and model_version:
            proposed_units = controller_module.compute_target_units(
                ticket=ticket,
                zscore=zscore,
                seed=seed,
                config=config,
                model_version=model_version,
                beta=beta,
                beta_stability=feat_row.get("beta_stability", 0.0),
                spread=feat_row.get("spread", 0.0),
                no_mean_cross_days=feat_row.get("no_mean_cross_days", 0),
                extreme_z=extreme_z,
            )
        else:
            proposed_units = controller_module.compute_target_units(
                ticket=ticket,
                zscore=zscore,
            )

        proposed_notional_a, proposed_notional_b = controller_module.compute_notionals(
            units=proposed_units,
            beta=beta,
            max_gross_notional=ticket.max_gross_notional,
            ticket=ticket,
        )

        # Apply per-ticket governor
        is_expired = ticket.is_expired(asof_date)
        governor_result = apply_risk_governor(
            ticket=ticket,
            beta=beta,
            proposed_units=proposed_units,
            proposed_notional_a=proposed_notional_a,
            proposed_notional_b=proposed_notional_b,
            extreme_z=extreme_z,
            is_expired=is_expired,
        )

        ticket_decisions.append({
            "ticket_id": ticket.ticket_id,
            "pair_id": pair_id,
            "leg_a": ticket.leg_a,
            "leg_b": ticket.leg_b,
            "action_units": governor_result.action_units,
            "executed_units": governor_result.executed_units,
            "executed_notional_a": governor_result.executed_notional_a,
            "executed_notional_b": governor_result.executed_notional_b,
            "clamp_codes": governor_result.clamp_codes,
        })

        logs.append(
            f"Ticket {ticket.ticket_id}: units={governor_result.executed_units:.2f}, "
            f"A={governor_result.executed_notional_a:,.0f}, B={governor_result.executed_notional_b:,.0f}"
        )

    # Step 4: Aggregate to symbol-level totals
    symbol_notionals, symbol_source_tickets = _aggregate_to_symbols(ticket_decisions)
    logs.append(f"Aggregated to {len(symbol_notionals)} symbols")

    # Step 5: Apply portfolio governor
    symbol_notionals, symbol_clamp_codes, portfolio_metrics = _apply_portfolio_governor(
        symbol_notionals=symbol_notionals,
        portfolio_max_gross_notional=portfolio_max_gross_notional,
        portfolio_max_name_gross_notional=portfolio_max_name_gross_notional,
        logs=logs,
    )

    # Step 6: Add CASH leg for exact net zero
    equity_net = sum(symbol_notionals.values())
    cash_notional = -equity_net
    symbol_notionals[CASH_SYMBOL] = cash_notional
    symbol_source_tickets[CASH_SYMBOL] = []

    if CASH_SYMBOL not in symbol_clamp_codes:
        symbol_clamp_codes[CASH_SYMBOL] = []
    symbol_clamp_codes[CASH_SYMBOL].append(CLAMP_NET_TO_CASH)

    logs.append(f"Added CASH leg: {cash_notional:,.0f} (net to zero)")

    # Verify final net is zero
    final_net = sum(symbol_notionals.values())
    if abs(final_net) > 0.01:
        raise ValueError(f"Final net not zero: {final_net}")

    # Build portfolio records
    portfolio_records = _build_portfolio_records(
        run_id=run_id,
        asof_date=asof_date,
        symbol_notionals=symbol_notionals,
        symbol_source_tickets=symbol_source_tickets,
        symbol_clamp_codes=symbol_clamp_codes,
    )

    # Write artifacts
    portfolio_df = _build_portfolio_dataframe(portfolio_records)
    _write_ideal_portfolio(run_dir, portfolio_df)
    _write_portfolio_blotter(run_dir, portfolio_records)

    # Compute final metrics
    equity_symbols = [s for s in symbol_notionals if s != CASH_SYMBOL]
    equity_gross = sum(abs(symbol_notionals[s]) for s in equity_symbols)
    equity_net_final = sum(symbol_notionals[s] for s in equity_symbols)

    metrics = {
        "equity_gross": equity_gross,
        "equity_net": equity_net_final,
        "cash_notional": cash_notional,
        "final_net": final_net,
        "name_caps_triggered_count": portfolio_metrics["name_caps_triggered"],
        "portfolio_gross_clamped": portfolio_metrics["portfolio_gross_clamped"],
        "total_symbols": len(symbol_notionals),
        "total_tickets": len(ticket_decisions),
    }
    _write_portfolio_metrics(run_dir, metrics)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "run_type": "ideal_portfolio",
        "asof_ts": datetime.utcnow().isoformat(),
        "asof_date": asof_date.isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "feature_version": feature_version,
        "controller_version": controller_version,
        "model_version": model_version,
        "portfolio_max_gross_notional": portfolio_max_gross_notional,
        "portfolio_max_name_gross_notional": portfolio_max_name_gross_notional,
        "net_mode": NET_MODE_EXACT_ZERO,
        "cash_symbol": CASH_SYMBOL,
        "tickets_snapshot_paths": snapshot_result.snapshot_paths,
        "ticket_hashes": snapshot_result.ticket_hashes,
        "fmp_cache_state_summary": get_fmp_cache_summary(config),
    }
    write_manifest(run_dir, manifest)

    logs.append(f"Portfolio run completed: {run_id}")
    write_logs(run_dir, "\n".join(logs))

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "asof_date": asof_date,
        "portfolio_records": portfolio_records,
        "ticket_decisions": ticket_decisions,
        "metrics": metrics,
    }


def _aggregate_to_symbols(
    ticket_decisions: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    # Aggregate ticket-level notionals to symbol-level
    symbol_notionals: Dict[str, float] = defaultdict(float)
    symbol_source_tickets: Dict[str, List[str]] = defaultdict(list)

    for dec in ticket_decisions:
        leg_a = dec["leg_a"]
        leg_b = dec["leg_b"]
        ticket_id = dec["ticket_id"]

        symbol_notionals[leg_a] += dec["executed_notional_a"]
        symbol_notionals[leg_b] += dec["executed_notional_b"]

        if ticket_id not in symbol_source_tickets[leg_a]:
            symbol_source_tickets[leg_a].append(ticket_id)
        if ticket_id not in symbol_source_tickets[leg_b]:
            symbol_source_tickets[leg_b].append(ticket_id)

    return dict(symbol_notionals), dict(symbol_source_tickets)


def _apply_portfolio_governor(
    symbol_notionals: Dict[str, float],
    portfolio_max_gross_notional: float,
    portfolio_max_name_gross_notional: float,
    logs: List[str],
) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, Any]]:
    # Apply portfolio-level constraints
    symbol_clamp_codes: Dict[str, List[str]] = defaultdict(list)
    name_caps_triggered = 0
    portfolio_gross_clamped = False

    # Step 1: Apply per-name gross cap
    for symbol in sorted(symbol_notionals.keys()):
        notional = symbol_notionals[symbol]
        if abs(notional) > portfolio_max_name_gross_notional:
            scale = portfolio_max_name_gross_notional / abs(notional)
            old_notional = notional
            symbol_notionals[symbol] = notional * scale
            symbol_clamp_codes[symbol].append(CLAMP_NAME_GROSS)
            name_caps_triggered += 1
            logs.append(
                f"Name cap: {symbol} scaled from {old_notional:,.0f} to {symbol_notionals[symbol]:,.0f}"
            )

    # Step 2: Apply portfolio gross cap
    portfolio_gross = sum(abs(n) for n in symbol_notionals.values())
    if portfolio_gross > portfolio_max_gross_notional:
        scale = portfolio_max_gross_notional / portfolio_gross
        portfolio_gross_clamped = True
        for symbol in symbol_notionals:
            symbol_notionals[symbol] = symbol_notionals[symbol] * scale
            symbol_clamp_codes[symbol].append(CLAMP_PORTFOLIO_GROSS)
        logs.append(f"Portfolio gross cap applied: scale={scale:.4f}")

    metrics = {
        "name_caps_triggered": name_caps_triggered,
        "portfolio_gross_clamped": portfolio_gross_clamped,
    }

    return symbol_notionals, dict(symbol_clamp_codes), metrics


def _build_portfolio_records(
    run_id: str,
    asof_date: date,
    symbol_notionals: Dict[str, float],
    symbol_source_tickets: Dict[str, List[str]],
    symbol_clamp_codes: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    records = []

    for symbol in sorted(symbol_notionals.keys()):
        notional = symbol_notionals[symbol]
        source_tickets = sorted(symbol_source_tickets.get(symbol, []))
        clamp_codes = sorted(symbol_clamp_codes.get(symbol, []))

        records.append({
            "run_id": run_id,
            "asof_date": asof_date,
            "symbol": symbol,
            "notional": notional,
            "source_ticket_ids": ",".join(source_tickets),
            "clamp_codes": ",".join(clamp_codes),
            "is_cash": 1 if symbol == CASH_SYMBOL else 0,
        })

    return records


def _build_portfolio_dataframe(records: List[Dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "asof_date": pl.Date,
                "symbol": pl.Utf8,
                "notional": pl.Float64,
                "source_ticket_ids": pl.Utf8,
                "clamp_codes": pl.Utf8,
                "is_cash": pl.Int8,
            }
        )

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
        pl.col("is_cash").cast(pl.Int8),
    ])

    column_order = [
        "run_id",
        "asof_date",
        "symbol",
        "notional",
        "source_ticket_ids",
        "clamp_codes",
        "is_cash",
    ]
    df = df.select(column_order)
    df = df.sort("symbol")

    return df


def _write_ideal_portfolio(run_dir: Path, df: pl.DataFrame) -> None:
    output_path = run_dir / "ideal_portfolio.parquet"
    df.write_parquet(output_path)


def _write_portfolio_blotter(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    output_path = run_dir / "portfolio_blotter.csv"

    if not records:
        with open(output_path, "w") as f:
            f.write("symbol,notional,source_ticket_ids,clamp_codes,is_cash\n")
        return

    df = pl.DataFrame(records)
    df = df.select(["symbol", "notional", "source_ticket_ids", "clamp_codes", "is_cash"])
    df = df.sort("symbol")
    df.write_csv(output_path)


def _write_portfolio_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    import json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str, sort_keys=True)


def _create_empty_portfolio_result(
    config: Config,
    run_dir: Path,
    run_id: str,
    asof_date: date,
    feature_version: str,
    logs: List[str],
) -> Dict[str, Any]:
    # Create an empty portfolio with just CASH = 0
    portfolio_records = [{
        "run_id": run_id,
        "asof_date": asof_date,
        "symbol": CASH_SYMBOL,
        "notional": 0.0,
        "source_ticket_ids": "",
        "clamp_codes": CLAMP_NET_TO_CASH,
        "is_cash": 1,
    }]

    portfolio_df = _build_portfolio_dataframe(portfolio_records)
    _write_ideal_portfolio(run_dir, portfolio_df)
    _write_portfolio_blotter(run_dir, portfolio_records)

    metrics = {
        "equity_gross": 0.0,
        "equity_net": 0.0,
        "cash_notional": 0.0,
        "final_net": 0.0,
        "name_caps_triggered_count": 0,
        "portfolio_gross_clamped": False,
        "total_symbols": 1,
        "total_tickets": 0,
    }
    _write_portfolio_metrics(run_dir, metrics)

    manifest = {
        "run_id": run_id,
        "run_type": "ideal_portfolio",
        "asof_ts": datetime.utcnow().isoformat(),
        "asof_date": asof_date.isoformat(),
        "git_sha": get_git_sha(),
        "seed": 0,
        "feature_version": feature_version,
        "controller_version": "none",
        "portfolio_max_gross_notional": 0,
        "portfolio_max_name_gross_notional": 0,
        "net_mode": NET_MODE_EXACT_ZERO,
        "cash_symbol": CASH_SYMBOL,
    }
    write_manifest(run_dir, manifest)

    write_logs(run_dir, "\n".join(logs))

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "asof_date": asof_date,
        "portfolio_records": portfolio_records,
        "ticket_decisions": [],
        "metrics": metrics,
    }


def get_latest_portfolio_run(config: Config, asof_date: date) -> Optional[Path]:
    # Find the most recent portfolio run for a given date
    if not config.runs_dir.exists():
        return None

    matching_runs = []
    for run_dir in config.runs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_id=portfolio_"):
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                import json
                with open(manifest_path) as f:
                    manifest = json.load(f)
                run_asof = manifest.get("asof_date")
                if run_asof == asof_date.isoformat():
                    matching_runs.append(run_dir)

    if not matching_runs:
        return None

    # Return the most recent (sorted by name which includes timestamp)
    return sorted(matching_runs, reverse=True)[0]
