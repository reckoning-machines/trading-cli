# Backtest Runner - runs historical backtests of portfolio strategies

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
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
    read_daily_bars,
    get_fmp_cache_summary,
)
from src.tickets import PairTicket, load_active_tickets, snapshot_tickets
from src.risk_governor import apply_risk_governor
from src import controller_baseline
from src import controller_rl_stub
from src.run_engine import get_git_sha
from src.portfolio_generator import (
    aggregate_ticket_decisions,
    apply_portfolio_governor,
    CLAMP_NET_TO_CASH,
)


DEFAULT_COST_BPS = 1.0


def generate_backtest_run_id(seed: int) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    hash_input = f"backtest_{ts}_{seed}".encode()
    short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    return f"backtest_{ts}_{short_hash}"


def get_trading_dates_in_range(
    config: Config,
    start_date: date,
    end_date: date,
    feature_version: str,
) -> List[date]:
    # Get all dates where we have feature data
    feature_base = config.pair_state_dir / f"version={feature_version}"
    if not feature_base.exists():
        return []

    available_dates = []
    for d in sorted(feature_base.iterdir()):
        if d.is_dir() and d.name.startswith("asof_date="):
            date_str = d.name.replace("asof_date=", "")
            try:
                dt = date.fromisoformat(date_str)
                if start_date <= dt <= end_date:
                    available_dates.append(dt)
            except ValueError:
                pass

    return sorted(available_dates)


def get_symbol_future_dates(
    config: Config,
    symbol: str,
    asof_date: date,
) -> Set[date]:
    # Get all dates > asof_date where we have price data for this symbol
    df = read_daily_bars(config, symbol)
    if df is None or df.is_empty():
        return set()

    future_dates = df.filter(pl.col("date") > asof_date).select("date").to_series().to_list()
    return set(future_dates)


def get_common_next_trading_date(
    config: Config,
    symbols: List[str],
    asof_date: date,
) -> Optional[date]:
    # Find the earliest date > asof_date that exists in ALL symbols
    # This mirrors the realized_pnl.py logic

    if not symbols:
        return None

    # Get future dates for each symbol
    symbol_futures = []
    for symbol in symbols:
        futures = get_symbol_future_dates(config, symbol, asof_date)
        if not futures:
            # If any symbol has no future dates, no common date exists
            return None
        symbol_futures.append(futures)

    # Compute intersection
    common_dates = symbol_futures[0]
    for futures in symbol_futures[1:]:
        common_dates = common_dates & futures

    if not common_dates:
        return None

    # Return the earliest common date
    return min(common_dates)


def get_price_on_date(
    config: Config,
    symbol: str,
    target_date: date,
) -> Optional[float]:
    if symbol == CASH_SYMBOL:
        return 1.0  # Cash has no price change

    df = read_daily_bars(config, symbol)
    if df is None or df.is_empty():
        return None

    row = df.filter(pl.col("date") == target_date)
    if row.is_empty():
        return None

    return float(row["adj_close"][0])


def run_backtest(
    config: Config,
    start_date: date,
    end_date: date,
    feature_version: str,
    controller_name: str,
    seed: int,
    portfolio_max_gross_notional: float,
    portfolio_max_name_gross_notional: float,
    include_previous_strategies: bool = False,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    # Validate constraints
    if portfolio_max_gross_notional <= 0:
        raise ValueError("portfolio_max_gross_notional must be > 0")
    if portfolio_max_name_gross_notional <= 0:
        raise ValueError("portfolio_max_name_gross_notional must be > 0")

    if include_previous_strategies:
        raise ValueError(
            "include_previous_strategies=yes is not implemented in v0. "
            "This would require loading tickets from tickets/archive with historical expires_on coverage."
        )

    ensure_directories(config)
    np.random.seed(seed)

    run_id = generate_backtest_run_id(seed)
    run_dir = create_run_directory(config, run_id)

    logs = []
    logs.append(f"Backtest run started: {run_id}")
    logs.append(f"Start date: {start_date}")
    logs.append(f"End date: {end_date}")
    logs.append(f"Feature version: {feature_version}")
    logs.append(f"Controller: {controller_name}")
    logs.append(f"Seed: {seed}")
    logs.append(f"Portfolio max gross: {portfolio_max_gross_notional:,.0f}")
    logs.append(f"Portfolio max name gross: {portfolio_max_name_gross_notional:,.0f}")
    logs.append(f"Include previous strategies: {include_previous_strategies}")

    # Load tickets as of end_date (today's tickets used across all dates)
    tickets = load_active_tickets(config, end_date)
    logs.append(f"Loaded {len(tickets)} tickets (as of {end_date})")

    if not tickets:
        logs.append("No active tickets - empty backtest")
        write_logs(run_dir, "\n".join(logs))
        raise ValueError("No active tickets found for backtest")

    # Snapshot tickets
    snapshot_result = snapshot_tickets(tickets, run_dir)

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

    # Get trading dates (days with feature data)
    trading_dates = get_trading_dates_in_range(config, start_date, end_date, feature_version)
    logs.append(f"Found {len(trading_dates)} trading dates with features")

    if not trading_dates:
        raise ValueError(f"No feature data available between {start_date} and {end_date}")

    # Tracking metrics for Change Set C
    total_trading_days = len(trading_dates)
    days_with_features = 0
    days_skipped_due_to_prices = 0
    days_with_pnl = 0

    # Run backtest for each day
    daily_results = []
    prev_symbol_notionals: Dict[str, float] = {}

    for t_date in trading_dates:
        day_result = _run_single_day(
            config=config,
            asof_date=t_date,
            feature_version=feature_version,
            controller_module=controller_module,
            controller_name=controller_name,
            tickets=tickets,
            portfolio_max_gross_notional=portfolio_max_gross_notional,
            portfolio_max_name_gross_notional=portfolio_max_name_gross_notional,
            prev_symbol_notionals=prev_symbol_notionals,
            model_version=model_version,
            seed=seed,
            logs=logs,
        )

        if day_result is None:
            # No features for this day (shouldn't happen given trading_dates, but handle anyway)
            continue

        days_with_features += 1

        if day_result.get("skipped_due_to_prices"):
            days_skipped_due_to_prices += 1
            # Do NOT add to daily_results - skip entirely
            continue

        days_with_pnl += 1
        daily_results.append(day_result)
        prev_symbol_notionals = day_result["symbol_notionals"]

    logs.append(f"Days with features: {days_with_features}")
    logs.append(f"Days skipped due to prices: {days_skipped_due_to_prices}")
    logs.append(f"Days with PnL: {days_with_pnl}")

    # Build timeseries dataframe
    timeseries_df = _build_timeseries_dataframe(daily_results, run_id)
    _write_backtest_timeseries(run_dir, timeseries_df)

    # Compute aggregate metrics with explicit skipped day tracking
    backtest_metrics = _compute_backtest_metrics(
        daily_results=daily_results,
        total_trading_days=total_trading_days,
        days_with_features=days_with_features,
        days_skipped_due_to_prices=days_skipped_due_to_prices,
        days_with_pnl=days_with_pnl,
    )
    _write_backtest_metrics(run_dir, backtest_metrics)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "run_type": "backtest",
        "asof_ts": datetime.utcnow().isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "feature_version": feature_version,
        "controller_version": controller_version,
        "model_version": model_version,
        "portfolio_max_gross_notional": portfolio_max_gross_notional,
        "portfolio_max_name_gross_notional": portfolio_max_name_gross_notional,
        "net_mode": NET_MODE_EXACT_ZERO,
        "cash_symbol": CASH_SYMBOL,
        "include_previous_strategies": include_previous_strategies,
        "tickets_snapshot_paths": snapshot_result.snapshot_paths,
        "ticket_hashes": snapshot_result.ticket_hashes,
        "fmp_cache_state_summary": get_fmp_cache_summary(config),
    }
    write_manifest(run_dir, manifest)

    logs.append(f"Backtest completed: {run_id}")
    write_logs(run_dir, "\n".join(logs))

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": total_trading_days,
        "daily_results": daily_results,
        "metrics": backtest_metrics,
    }


def _run_single_day(
    config: Config,
    asof_date: date,
    feature_version: str,
    controller_module,
    controller_name: str,
    tickets: List[PairTicket],
    portfolio_max_gross_notional: float,
    portfolio_max_name_gross_notional: float,
    prev_symbol_notionals: Dict[str, float],
    model_version: Optional[str],
    seed: int,
    logs: List[str],
) -> Optional[Dict[str, Any]]:
    # Load features
    features_df = read_pair_state_features(config, feature_version, asof_date)
    if features_df is None:
        logs.append(f"  {asof_date}: No features, skipping")
        return None

    # Compute per-ticket decisions
    ticket_decisions = []

    for ticket in tickets:
        # Skip expired tickets for this date
        if ticket.is_expired(asof_date):
            continue

        pair_id = ticket.pair_id

        pair_features = features_df.filter(pl.col("pair_id") == pair_id)
        if pair_features.is_empty():
            continue

        # Change Set D: Enforce feature row uniqueness in backtest
        if pair_features.height > 1:
            raise ValueError(
                f"Multiple feature rows for pair {pair_id} on {asof_date}: "
                f"expected 1, got {pair_features.height}"
            )

        feat_row = pair_features.row(0, named=True)
        beta = feat_row["beta"]
        zscore = feat_row["zscore"]
        extreme_z = feat_row["extreme_z"]

        if not np.isfinite(beta) or not np.isfinite(zscore):
            continue

        # Compute action
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
        governor_result = apply_risk_governor(
            ticket=ticket,
            beta=beta,
            proposed_units=proposed_units,
            proposed_notional_a=proposed_notional_a,
            proposed_notional_b=proposed_notional_b,
            extreme_z=extreme_z,
            is_expired=False,
        )

        ticket_decisions.append({
            "ticket_id": ticket.ticket_id,
            "pair_id": pair_id,
            "leg_a": ticket.leg_a,
            "leg_b": ticket.leg_b,
            "executed_notional_a": governor_result.executed_notional_a,
            "executed_notional_b": governor_result.executed_notional_b,
        })

    # Aggregate to symbols using shared function
    contributions_by_symbol, symbol_source_tickets = aggregate_ticket_decisions(ticket_decisions)

    # Apply portfolio governor using shared function (same logic as ideal portfolio)
    governor_result = apply_portfolio_governor(
        contributions_by_symbol=contributions_by_symbol,
        symbol_source_tickets=symbol_source_tickets,
        portfolio_max_gross_notional=portfolio_max_gross_notional,
        portfolio_max_name_gross_notional=portfolio_max_name_gross_notional,
        logs=[],  # Don't clutter main logs with per-day governor details
    )

    symbol_notionals = governor_result.symbol_net

    # Add CASH leg (CASH does not count toward equity gross caps)
    equity_net = sum(symbol_notionals.values())
    cash_notional = -equity_net
    symbol_notionals[CASH_SYMBOL] = cash_notional

    # Compute gross (explicitly excludes CASH)
    equity_symbols = sorted([s for s in symbol_notionals if s != CASH_SYMBOL])
    equity_gross = sum(abs(symbol_notionals[s]) for s in equity_symbols)

    # Get only equity symbols with non-trivial notionals for common next date
    nontrivial_symbols = [s for s in equity_symbols if abs(symbol_notionals[s]) >= 0.01]

    # Change Set B: Find common next trading date across ALL equity symbols
    if nontrivial_symbols:
        common_next_date = get_common_next_trading_date(config, nontrivial_symbols, asof_date)
    else:
        common_next_date = None

    # If no common next date exists, skip this day entirely
    if common_next_date is None and nontrivial_symbols:
        logs.append(f"  {asof_date}: No common next trading date for equity symbols, skipping PnL")
        return {
            "asof_date": asof_date,
            "symbol_notionals": symbol_notionals.copy(),
            "skipped_due_to_prices": True,
        }

    # Compute turnover (change from previous day)
    turnover = 0.0
    all_symbols = set(symbol_notionals.keys()) | set(prev_symbol_notionals.keys())
    for symbol in sorted(all_symbols):
        curr = symbol_notionals.get(symbol, 0.0)
        prev = prev_symbol_notionals.get(symbol, 0.0)
        turnover += abs(curr - prev)

    # Compute costs
    costs = (DEFAULT_COST_BPS / 10000.0) * turnover

    # Compute PnL using common next date for ALL equity symbols
    pnl_gross = 0.0

    if common_next_date is not None:
        for symbol in equity_symbols:
            notional = symbol_notionals[symbol]
            if abs(notional) < 0.01:
                continue

            price_today = get_price_on_date(config, symbol, asof_date)
            price_next = get_price_on_date(config, symbol, common_next_date)

            if price_today is None or price_next is None:
                # This shouldn't happen if common_next_date is valid, but handle defensively
                continue

            ret = (price_next / price_today) - 1.0
            pnl_gross += notional * ret

    pnl_net = pnl_gross - costs

    logs.append(
        f"  {asof_date}: gross={equity_gross:,.0f}, turnover={turnover:,.0f}, "
        f"pnl_gross={pnl_gross:,.0f}, pnl_net={pnl_net:,.0f}, next_date={common_next_date}"
    )

    return {
        "asof_date": asof_date,
        "next_date": common_next_date,
        "symbol_notionals": symbol_notionals.copy(),
        "equity_gross": equity_gross,
        "equity_net": sum(symbol_notionals[s] for s in equity_symbols),
        "cash_notional": cash_notional,
        "turnover": turnover,
        "costs": costs,
        "pnl_gross": pnl_gross,
        "pnl_net": pnl_net,
        "skipped_due_to_prices": False,
    }


def _build_timeseries_dataframe(
    daily_results: List[Dict[str, Any]],
    run_id: str,
) -> pl.DataFrame:
    if not daily_results:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "asof_date": pl.Date,
                "next_date": pl.Date,
                "equity_gross": pl.Float64,
                "equity_net": pl.Float64,
                "cash_notional": pl.Float64,
                "turnover": pl.Float64,
                "costs": pl.Float64,
                "pnl_gross": pl.Float64,
                "pnl_net": pl.Float64,
                "cumulative_pnl_net": pl.Float64,
            }
        )

    records = []
    cumulative_pnl = 0.0

    for result in daily_results:
        cumulative_pnl += result["pnl_net"]
        records.append({
            "run_id": run_id,
            "asof_date": result["asof_date"],
            "next_date": result.get("next_date"),
            "equity_gross": result["equity_gross"],
            "equity_net": result["equity_net"],
            "cash_notional": result["cash_notional"],
            "turnover": result["turnover"],
            "costs": result["costs"],
            "pnl_gross": result["pnl_gross"],
            "pnl_net": result["pnl_net"],
            "cumulative_pnl_net": cumulative_pnl,
        })

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
        pl.col("next_date").cast(pl.Date),
    ])

    return df


def _write_backtest_timeseries(run_dir: Path, df: pl.DataFrame) -> None:
    output_path = run_dir / "backtest_timeseries.parquet"
    df.write_parquet(output_path)


def _compute_backtest_metrics(
    daily_results: List[Dict[str, Any]],
    total_trading_days: int,
    days_with_features: int,
    days_skipped_due_to_prices: int,
    days_with_pnl: int,
) -> Dict[str, Any]:
    # Change Set C: Include explicit tracking of skipped days

    if not daily_results:
        return {
            "total_trading_days": total_trading_days,
            "days_with_features": days_with_features,
            "days_skipped_due_to_prices": days_skipped_due_to_prices,
            "days_with_pnl": days_with_pnl,
            "total_pnl_gross": 0.0,
            "total_pnl_net": 0.0,
            "total_costs": 0.0,
            "total_turnover": 0.0,
            "avg_daily_pnl_net": 0.0,
            "std_daily_pnl_net": 0.0,
            "sharpe_ratio": 0.0,
        }

    total_pnl_gross = sum(r["pnl_gross"] for r in daily_results)
    total_pnl_net = sum(r["pnl_net"] for r in daily_results)
    total_costs = sum(r["costs"] for r in daily_results)
    total_turnover = sum(r["turnover"] for r in daily_results)

    # Sharpe and averages computed only over days_with_pnl
    pnl_series = [r["pnl_net"] for r in daily_results]
    avg_daily_pnl = np.mean(pnl_series) if pnl_series else 0.0
    std_daily_pnl = np.std(pnl_series) if len(pnl_series) > 1 else 0.0

    # Annualized Sharpe (assuming 252 trading days)
    sharpe = 0.0
    if std_daily_pnl > 0:
        sharpe = (avg_daily_pnl / std_daily_pnl) * np.sqrt(252)

    return {
        "total_trading_days": total_trading_days,
        "days_with_features": days_with_features,
        "days_skipped_due_to_prices": days_skipped_due_to_prices,
        "days_with_pnl": days_with_pnl,
        "total_pnl_gross": float(total_pnl_gross),
        "total_pnl_net": float(total_pnl_net),
        "total_costs": float(total_costs),
        "total_turnover": float(total_turnover),
        "avg_daily_pnl_net": float(avg_daily_pnl),
        "std_daily_pnl_net": float(std_daily_pnl),
        "sharpe_ratio": float(sharpe),
    }


def _write_backtest_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    import json
    metrics_path = run_dir / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str, sort_keys=True)
