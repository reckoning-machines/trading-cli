# Realized PnL Label Generation - computes daily PnL and turnover-based costs

from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import polars as pl

from src.config import Config
from src.storage import read_daily_bars, read_decisions


STATUS_OK = "OK"
STATUS_NO_PRICE_SKIP = "NO_PRICE_SKIP"


def get_next_trading_date(
    config: Config,
    symbol: str,
    asof_date: date,
) -> Optional[date]:
    df = read_daily_bars(config, symbol)
    if df is None or df.is_empty():
        return None

    future_dates = df.filter(pl.col("date") > asof_date).sort("date")
    if future_dates.is_empty():
        return None

    return future_dates["date"][0]


def get_price_on_date(
    config: Config,
    symbol: str,
    target_date: date,
) -> Optional[float]:
    df = read_daily_bars(config, symbol)
    if df is None or df.is_empty():
        return None

    row = df.filter(pl.col("date") == target_date)
    if row.is_empty():
        return None

    return float(row["adj_close"][0])


def read_last_positions(config: Config) -> Dict[str, Tuple[float, float, date]]:
    positions_path = config.last_positions_dir / "positions.parquet"
    if not positions_path.exists():
        return {}

    df = pl.read_parquet(positions_path)
    result = {}
    for row in df.iter_rows(named=True):
        result[row["ticket_id"]] = (
            row["last_notional_a"],
            row["last_notional_b"],
            row["last_asof_date"],
        )
    return result


def write_last_positions(
    config: Config,
    positions: Dict[str, Tuple[float, float, date]],
) -> None:
    config.last_positions_dir.mkdir(parents=True, exist_ok=True)
    positions_path = config.last_positions_dir / "positions.parquet"

    records = []
    for ticket_id in sorted(positions.keys()):
        notional_a, notional_b, asof_date = positions[ticket_id]
        records.append({
            "ticket_id": ticket_id,
            "last_notional_a": notional_a,
            "last_notional_b": notional_b,
            "last_asof_date": asof_date,
        })

    if not records:
        df = pl.DataFrame(
            schema={
                "ticket_id": pl.Utf8,
                "last_notional_a": pl.Float64,
                "last_notional_b": pl.Float64,
                "last_asof_date": pl.Date,
            }
        )
    else:
        df = pl.DataFrame(records)
        df = df.with_columns([
            pl.col("ticket_id").cast(pl.Utf8),
            pl.col("last_notional_a").cast(pl.Float64),
            pl.col("last_notional_b").cast(pl.Float64),
            pl.col("last_asof_date").cast(pl.Date),
        ])

    df.write_parquet(positions_path)


def write_realized_to_run(run_dir: Path, df: pl.DataFrame) -> None:
    output_path = run_dir / "realized.parquet"
    df.write_parquet(output_path)


def write_realized_to_labels(
    config: Config,
    label_version: str,
    asof_date: date,
    df: pl.DataFrame,
) -> None:
    version_dir = config.realized_labels_dir / f"version={label_version}"
    date_dir = version_dir / f"asof_date={asof_date.isoformat()}"
    date_dir.mkdir(parents=True, exist_ok=True)
    output_path = date_dir / "part-0000.parquet"
    df.write_parquet(output_path)


def get_leg_symbols_from_blotter(run_dir: Path) -> Dict[str, Tuple[str, str]]:
    blotter_path = run_dir / "blotter.csv"
    if not blotter_path.exists():
        return {}

    df = pl.read_csv(blotter_path)
    result = {}
    for row in df.iter_rows(named=True):
        result[row["ticket_id"]] = (row["leg_a"], row["leg_b"])
    return result


def compute_realized_pnl(
    config: Config,
    run_id: str,
    cost_bps: float,
    label_version: str,
) -> Dict[str, Any]:
    run_dir = config.runs_dir / f"run_id={run_id}"

    if not run_dir.exists():
        for d in config.runs_dir.iterdir():
            if d.is_dir() and run_id in d.name:
                run_dir = d
                break

    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_id}")

    decisions_df = read_decisions(run_dir)
    if decisions_df is None or decisions_df.is_empty():
        raise ValueError(f"No decisions found in run: {run_id}")

    leg_symbols = get_leg_symbols_from_blotter(run_dir)

    last_positions = read_last_positions(config)

    asof_date = decisions_df["asof_date"][0]

    realized_records = []
    new_positions = {}

    decisions_sorted = decisions_df.sort("ticket_id")

    for row in decisions_sorted.iter_rows(named=True):
        ticket_id = row["ticket_id"]
        pair_id = row["pair_id"]
        executed_notional_a = row["executed_notional_a"]
        executed_notional_b = row["executed_notional_b"]

        if ticket_id in leg_symbols:
            leg_a, leg_b = leg_symbols[ticket_id]
        else:
            parts = pair_id.split("__")
            if len(parts) == 2:
                leg_a, leg_b = parts
            else:
                leg_a, leg_b = pair_id, pair_id

        price_a_asof = get_price_on_date(config, leg_a, asof_date)
        price_b_asof = get_price_on_date(config, leg_b, asof_date)

        next_date_a = get_next_trading_date(config, leg_a, asof_date)
        next_date_b = get_next_trading_date(config, leg_b, asof_date)

        if next_date_a is None or next_date_b is None:
            next_date = None
        else:
            next_date = max(next_date_a, next_date_b)

        if next_date is not None:
            price_a_next = get_price_on_date(config, leg_a, next_date)
            price_b_next = get_price_on_date(config, leg_b, next_date)
        else:
            price_a_next = None
            price_b_next = None

        if ticket_id in last_positions:
            prev_notional_a, prev_notional_b, _ = last_positions[ticket_id]
        else:
            prev_notional_a = 0.0
            prev_notional_b = 0.0

        trade_notional = (
            abs(executed_notional_a - prev_notional_a) +
            abs(executed_notional_b - prev_notional_b)
        )
        costs = (cost_bps / 10000.0) * trade_notional

        gross_exposure = abs(executed_notional_a) + abs(executed_notional_b)

        if (price_a_asof is None or price_b_asof is None or
            price_a_next is None or price_b_next is None or
            next_date is None):
            realized_records.append({
                "run_id": run_id,
                "asof_date": asof_date,
                "next_date": next_date,
                "ticket_id": ticket_id,
                "pair_id": pair_id,
                "leg_a": leg_a,
                "leg_b": leg_b,
                "executed_notional_a": executed_notional_a,
                "executed_notional_b": executed_notional_b,
                "prev_notional_a": prev_notional_a,
                "prev_notional_b": prev_notional_b,
                "trade_notional": trade_notional,
                "gross_exposure": gross_exposure,
                "ret_a": 0.0,
                "ret_b": 0.0,
                "pnl_gross": 0.0,
                "costs": 0.0,
                "pnl_net": 0.0,
                "cost_bps": cost_bps,
                "label_version": label_version,
                "status": STATUS_NO_PRICE_SKIP,
            })
        else:
            ret_a = (price_a_next / price_a_asof) - 1.0
            ret_b = (price_b_next / price_b_asof) - 1.0

            pnl_gross = (executed_notional_a * ret_a) + (executed_notional_b * ret_b)
            pnl_net = pnl_gross - costs

            realized_records.append({
                "run_id": run_id,
                "asof_date": asof_date,
                "next_date": next_date,
                "ticket_id": ticket_id,
                "pair_id": pair_id,
                "leg_a": leg_a,
                "leg_b": leg_b,
                "executed_notional_a": executed_notional_a,
                "executed_notional_b": executed_notional_b,
                "prev_notional_a": prev_notional_a,
                "prev_notional_b": prev_notional_b,
                "trade_notional": trade_notional,
                "gross_exposure": gross_exposure,
                "ret_a": ret_a,
                "ret_b": ret_b,
                "pnl_gross": pnl_gross,
                "costs": costs,
                "pnl_net": pnl_net,
                "cost_bps": cost_bps,
                "label_version": label_version,
                "status": STATUS_OK,
            })

        new_positions[ticket_id] = (executed_notional_a, executed_notional_b, asof_date)

    realized_df = build_realized_dataframe(realized_records)

    validation_errors = validate_realized(realized_df, cost_bps)

    write_realized_to_run(run_dir, realized_df)

    write_realized_to_labels(config, label_version, asof_date, realized_df)

    write_last_positions(config, new_positions)

    summary = compute_realized_summary(realized_df)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "asof_date": asof_date,
        "realized_records": realized_records,
        "summary": summary,
        "validation_errors": validation_errors,
    }


def build_realized_dataframe(records: List[Dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "asof_date": pl.Date,
                "next_date": pl.Date,
                "ticket_id": pl.Utf8,
                "pair_id": pl.Utf8,
                "leg_a": pl.Utf8,
                "leg_b": pl.Utf8,
                "executed_notional_a": pl.Float64,
                "executed_notional_b": pl.Float64,
                "prev_notional_a": pl.Float64,
                "prev_notional_b": pl.Float64,
                "trade_notional": pl.Float64,
                "gross_exposure": pl.Float64,
                "ret_a": pl.Float64,
                "ret_b": pl.Float64,
                "pnl_gross": pl.Float64,
                "costs": pl.Float64,
                "pnl_net": pl.Float64,
                "cost_bps": pl.Float64,
                "label_version": pl.Utf8,
                "status": pl.Utf8,
            }
        )

    column_order = [
        "run_id",
        "asof_date",
        "next_date",
        "ticket_id",
        "pair_id",
        "leg_a",
        "leg_b",
        "executed_notional_a",
        "executed_notional_b",
        "prev_notional_a",
        "prev_notional_b",
        "trade_notional",
        "gross_exposure",
        "ret_a",
        "ret_b",
        "pnl_gross",
        "costs",
        "pnl_net",
        "cost_bps",
        "label_version",
        "status",
    ]

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
        pl.col("next_date").cast(pl.Date),
    ])

    df = df.select(column_order)

    df = df.sort("ticket_id")

    return df


def validate_realized(df: pl.DataFrame, cost_bps: float) -> List[str]:
    errors = []

    ok_rows = df.filter(pl.col("status") == STATUS_OK)
    for row in ok_rows.iter_rows(named=True):
        if row["next_date"] is not None and row["asof_date"] is not None:
            if row["next_date"] <= row["asof_date"]:
                errors.append(f"Ticket {row['ticket_id']}: next_date not > asof_date")

        if row["trade_notional"] < 0:
            errors.append(f"Ticket {row['ticket_id']}: negative trade_notional")

        expected_costs = (cost_bps / 10000.0) * row["trade_notional"]
        if abs(row["costs"] - expected_costs) > 0.01:
            errors.append(f"Ticket {row['ticket_id']}: costs mismatch")

    skip_rows = df.filter(pl.col("status") == STATUS_NO_PRICE_SKIP)
    for row in skip_rows.iter_rows(named=True):
        if row["pnl_gross"] != 0.0:
            errors.append(f"Ticket {row['ticket_id']}: pnl_gross not 0 for NO_PRICE_SKIP")
        if row["costs"] != 0.0:
            errors.append(f"Ticket {row['ticket_id']}: costs not 0 for NO_PRICE_SKIP")
        if row["pnl_net"] != 0.0:
            errors.append(f"Ticket {row['ticket_id']}: pnl_net not 0 for NO_PRICE_SKIP")

    return errors


def compute_realized_summary(df: pl.DataFrame) -> Dict[str, Any]:
    ok_count = df.filter(pl.col("status") == STATUS_OK).height
    skip_count = df.filter(pl.col("status") == STATUS_NO_PRICE_SKIP).height

    total_pnl_net = df.select(pl.col("pnl_net").sum()).item() or 0.0
    total_costs = df.select(pl.col("costs").sum()).item() or 0.0
    total_trade_notional = df.select(pl.col("trade_notional").sum()).item() or 0.0
    total_pnl_gross = df.select(pl.col("pnl_gross").sum()).item() or 0.0

    return {
        "count_ok": ok_count,
        "count_no_price_skip": skip_count,
        "total_pnl_gross": total_pnl_gross,
        "total_pnl_net": total_pnl_net,
        "total_costs": total_costs,
        "total_trade_notional": total_trade_notional,
    }
