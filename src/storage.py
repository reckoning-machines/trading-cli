# Storage module - handles Parquet dataset operations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import polars as pl

from src.config import Config
from src.fmp_client import compute_data_checksum


def ensure_directories(config: Config) -> None:
    config.daily_bars_dir.mkdir(parents=True, exist_ok=True)
    config.fmp_meta_dir.mkdir(parents=True, exist_ok=True)
    config.pair_state_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.tickets_dir.mkdir(parents=True, exist_ok=True)
    Path(config.tickets_dir).parent.joinpath("archived").mkdir(parents=True, exist_ok=True)
    config.realized_labels_dir.mkdir(parents=True, exist_ok=True)
    config.last_positions_dir.mkdir(parents=True, exist_ok=True)


def write_daily_bars(config: Config, symbol: str, df: pl.DataFrame) -> None:
    symbol_dir = config.daily_bars_dir / f"symbol={symbol.upper()}"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    output_path = symbol_dir / "part-0000.parquet"

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        combined = pl.concat([existing, df])
        combined = combined.unique(subset=["date", "symbol"]).sort("date")
        combined.write_parquet(output_path)
    else:
        df.write_parquet(output_path)


def read_daily_bars(config: Config, symbol: str) -> Optional[pl.DataFrame]:
    symbol_dir = config.daily_bars_dir / f"symbol={symbol.upper()}"
    parquet_path = symbol_dir / "part-0000.parquet"

    if not parquet_path.exists():
        return None

    return pl.read_parquet(parquet_path)


def read_all_daily_bars(config: Config) -> pl.DataFrame:
    if not config.daily_bars_dir.exists():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "adj_close": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

    frames = []
    # Fix 4: Iterate over sorted directories for deterministic ordering
    symbol_dirs = sorted([
        d for d in config.daily_bars_dir.iterdir()
        if d.is_dir() and d.name.startswith("symbol=")
    ], key=lambda d: d.name)

    for symbol_dir in symbol_dirs:
        parquet_path = symbol_dir / "part-0000.parquet"
        if parquet_path.exists():
            frames.append(pl.read_parquet(parquet_path))

    if not frames:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "adj_close": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

    return pl.concat(frames).sort(["symbol", "date"])


def write_pull_log(
    config: Config,
    symbol: str,
    start_date: date,
    end_date: date,
    row_count: int,
    data_checksum: str,
) -> None:
    log_path = config.fmp_meta_dir / "fmp_pull_log.parquet"

    new_row = pl.DataFrame({
        "pulled_at_ts": [datetime.utcnow()],
        "symbol": [symbol.upper()],
        "start_date": [start_date],
        "end_date": [end_date],
        "row_count": [row_count],
        "data_checksum": [data_checksum],
        "source": ["FMP"],
    })

    new_row = new_row.with_columns([
        pl.col("pulled_at_ts").cast(pl.Datetime),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("start_date").cast(pl.Date),
        pl.col("end_date").cast(pl.Date),
        pl.col("row_count").cast(pl.Int64),
        pl.col("data_checksum").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
    ])

    if log_path.exists():
        existing = pl.read_parquet(log_path)
        combined = pl.concat([existing, new_row])
        combined.write_parquet(log_path)
    else:
        config.fmp_meta_dir.mkdir(parents=True, exist_ok=True)
        new_row.write_parquet(log_path)


def read_pull_log(config: Config) -> Optional[pl.DataFrame]:
    log_path = config.fmp_meta_dir / "fmp_pull_log.parquet"
    if not log_path.exists():
        return None
    return pl.read_parquet(log_path)


def write_pair_state_features(
    config: Config,
    feature_version: str,
    asof_date: date,
    df: pl.DataFrame,
) -> None:
    version_dir = config.pair_state_dir / f"version={feature_version}"
    date_dir = version_dir / f"asof_date={asof_date.isoformat()}"
    date_dir.mkdir(parents=True, exist_ok=True)
    output_path = date_dir / "part-0000.parquet"
    df.write_parquet(output_path)


def read_pair_state_features(
    config: Config,
    feature_version: str,
    asof_date: date,
) -> Optional[pl.DataFrame]:
    version_dir = config.pair_state_dir / f"version={feature_version}"
    date_dir = version_dir / f"asof_date={asof_date.isoformat()}"
    parquet_path = date_dir / "part-0000.parquet"

    if not parquet_path.exists():
        return None

    return pl.read_parquet(parquet_path)


def get_fmp_cache_summary(config: Config) -> Dict[str, str]:
    # Fix 4: Return dict with sorted keys for deterministic ordering
    summary = {}
    if not config.daily_bars_dir.exists():
        return summary

    # Iterate over sorted directories
    symbol_dirs = sorted([
        d for d in config.daily_bars_dir.iterdir()
        if d.is_dir() and d.name.startswith("symbol=")
    ], key=lambda d: d.name)

    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name.replace("symbol=", "")
        parquet_path = symbol_dir / "part-0000.parquet"
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            if not df.is_empty():
                max_date = df.select(pl.col("date").max()).item()
                summary[symbol] = str(max_date)

    # Return with sorted keys (dict maintains insertion order in Python 3.7+)
    return dict(sorted(summary.items()))


def create_run_directory(config: Config, run_id: str) -> Path:
    run_dir = config.runs_dir / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tickets_snapshot_dir = run_dir / "tickets_snapshot"
    tickets_snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str, sort_keys=True)


def write_decisions(run_dir: Path, df: pl.DataFrame) -> None:
    output_path = run_dir / "decisions.parquet"
    df.write_parquet(output_path)


def read_decisions(run_dir: Path) -> Optional[pl.DataFrame]:
    parquet_path = run_dir / "decisions.parquet"
    if not parquet_path.exists():
        return None
    return pl.read_parquet(parquet_path)


def write_blotter(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    output_path = run_dir / "blotter.csv"
    if not records:
        with open(output_path, "w") as f:
            f.write("ticket_id,pair_id,leg_a,leg_b,action_units,executed_units,executed_notional_a,executed_notional_b,clamp_codes\n")
        return

    df = pl.DataFrame(records)
    df.write_csv(output_path)


def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str, sort_keys=True)


def read_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def write_logs(run_dir: Path, logs: str) -> None:
    logs_path = run_dir / "logs.txt"
    with open(logs_path, "w") as f:
        f.write(logs)
