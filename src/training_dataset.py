# Training Dataset Builder - constructs training examples for RL policy

from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import polars as pl
import numpy as np

from src.config import Config
from src.storage import read_pair_state_features


ALLOWED_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def get_available_dates(
    config: Config,
    feature_version: str,
    label_version: str,
) -> Tuple[Set[date], Set[date]]:
    feature_dates = set()
    label_dates = set()

    feature_base = config.pair_state_dir / f"version={feature_version}"
    if feature_base.exists():
        for d in sorted(feature_base.iterdir()):
            if d.is_dir() and d.name.startswith("asof_date="):
                date_str = d.name.replace("asof_date=", "")
                try:
                    feature_dates.add(date.fromisoformat(date_str))
                except ValueError:
                    pass

    label_base = config.realized_labels_dir / f"version={label_version}"
    if label_base.exists():
        for d in sorted(label_base.iterdir()):
            if d.is_dir() and d.name.startswith("asof_date="):
                date_str = d.name.replace("asof_date=", "")
                try:
                    label_dates.add(date.fromisoformat(date_str))
                except ValueError:
                    pass

    return feature_dates, label_dates


def read_realized_labels(
    config: Config,
    label_version: str,
    asof_date: date,
) -> Optional[pl.DataFrame]:
    version_dir = config.realized_labels_dir / f"version={label_version}"
    date_dir = version_dir / f"asof_date={asof_date.isoformat()}"
    parquet_path = date_dir / "part-0000.parquet"

    if not parquet_path.exists():
        return None

    return pl.read_parquet(parquet_path)


def get_future_pnl_for_pair(
    config: Config,
    label_version: str,
    pair_id: str,
    start_date: date,
    horizon_days: int = 5,
) -> Optional[float]:
    # Get the next horizon_days of realized labels for this pair
    # and sum up the pnl_net values
    label_base = config.realized_labels_dir / f"version={label_version}"
    if not label_base.exists():
        return None

    # Get all available label dates after start_date
    future_dates = []
    for d in sorted(label_base.iterdir()):
        if d.is_dir() and d.name.startswith("asof_date="):
            date_str = d.name.replace("asof_date=", "")
            try:
                label_date = date.fromisoformat(date_str)
                if label_date > start_date:
                    future_dates.append(label_date)
            except ValueError:
                pass

    future_dates = sorted(future_dates)[:horizon_days]

    if not future_dates:
        return None

    total_pnl = 0.0
    for label_date in future_dates:
        labels_df = read_realized_labels(config, label_version, label_date)
        if labels_df is None:
            continue

        pair_row = labels_df.filter(pl.col("pair_id") == pair_id)
        if pair_row.is_empty():
            continue

        # Sum pnl_net for this pair on this date
        pnl = pair_row.select(pl.col("pnl_net").sum()).item()
        if pnl is not None:
            total_pnl += pnl

    return total_pnl


def build_training_examples(
    config: Config,
    start_date: date,
    end_date: date,
    feature_version: str,
    label_version: str,
    reward_horizon: int = 5,
) -> pl.DataFrame:
    # Build training examples from features and realized labels
    # Each example consists of:
    # - State features (from pair_state)
    # - Ticket conditioning features
    # - Action taken (from decisions/realized labels)
    # - 5-day reward (sum of pnl_net over next 5 trading days)

    feature_dates, label_dates = get_available_dates(config, feature_version, label_version)

    # Only use dates where we have both features and labels
    common_dates = sorted(feature_dates & label_dates)
    common_dates = [d for d in common_dates if start_date <= d <= end_date]

    if not common_dates:
        return _empty_training_dataframe()

    records = []

    for asof_date in common_dates:
        features_df = read_pair_state_features(config, feature_version, asof_date)
        labels_df = read_realized_labels(config, label_version, asof_date)

        if features_df is None or labels_df is None:
            continue

        # Join features with labels on pair_id
        for label_row in labels_df.iter_rows(named=True):
            pair_id = label_row["pair_id"]
            ticket_id = label_row["ticket_id"]

            # Get feature row for this pair
            feature_row = features_df.filter(pl.col("pair_id") == pair_id)
            if feature_row.is_empty():
                continue

            feat = feature_row.row(0, named=True)

            # Compute 5-day forward reward
            reward_5d = get_future_pnl_for_pair(
                config, label_version, pair_id, asof_date, reward_horizon
            )

            if reward_5d is None:
                # Skip examples where we cannot compute the reward
                continue

            # Extract executed_notional values to infer action taken
            executed_notional_a = label_row.get("executed_notional_a", 0.0)
            executed_notional_b = label_row.get("executed_notional_b", 0.0)
            gross_exposure = label_row.get("gross_exposure", 0.0)

            # Infer action_units from the executed position
            # This is approximate - we use the gross exposure ratio
            if gross_exposure > 0:
                # Try to map back to discrete action
                # Action units are in [-1, -0.5, 0, 0.5, 1]
                # We need ticket info to properly map back
                # For now, use sign of notional_a and magnitude from gross
                sign = 1.0 if executed_notional_a >= 0 else -1.0
                # Approximate action magnitude
                action_units = sign * min(1.0, gross_exposure / 100000.0)  # Rough scaling
                # Snap to allowed actions
                action_units = _snap_to_allowed_action(action_units)
            else:
                action_units = 0.0

            # Build the training record
            records.append({
                "asof_date": asof_date,
                "pair_id": pair_id,
                "ticket_id": ticket_id,
                # State features
                "zscore": feat.get("zscore", 0.0),
                "beta": feat.get("beta", 1.0),
                "beta_stability": feat.get("beta_stability", 0.0),
                "spread": feat.get("spread", 0.0),
                "no_mean_cross_days": feat.get("no_mean_cross_days", 0),
                "extreme_z": feat.get("extreme_z", 0),
                # Ticket conditioning features (from labels if available)
                "conviction_scaled": _scale_conviction(label_row.get("conviction", 3)),
                "max_gross_notional_scaled": _scale_notional(
                    label_row.get("max_gross_notional", 1000000.0)
                ),
                "orientation_long_spread": 1 if label_row.get("orientation", "LONG_SPREAD") == "LONG_SPREAD" else 0,
                "orientation_short_spread": 1 if label_row.get("orientation", "LONG_SPREAD") == "SHORT_SPREAD" else 0,
                "horizon_scaled": _scale_horizon(label_row.get("horizon_days", 30)),
                "time_stop_scaled": _scale_time_stop(label_row.get("time_stop_days", 20)),
                # Action and reward
                "action_units": action_units,
                "reward_5d": reward_5d,
                # Metadata
                "feature_version": feature_version,
                "label_version": label_version,
            })

    if not records:
        return _empty_training_dataframe()

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
    ])
    df = df.sort(["asof_date", "pair_id"])

    return df


def build_training_examples_from_decisions(
    config: Config,
    start_date: date,
    end_date: date,
    feature_version: str,
    label_version: str,
    reward_horizon: int = 5,
) -> pl.DataFrame:
    # Alternative builder that uses decisions.parquet from runs
    # This gives us the actual action_units taken by the controller

    feature_dates, label_dates = get_available_dates(config, feature_version, label_version)
    common_dates = sorted(feature_dates & label_dates)
    common_dates = [d for d in common_dates if start_date <= d <= end_date]

    if not common_dates:
        return _empty_training_dataframe()

    # Find runs for each date
    run_dirs_by_date = {}
    if config.runs_dir.exists():
        for run_dir in sorted(config.runs_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run_id="):
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    import json
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    run_asof = manifest.get("asof_date")
                    if run_asof:
                        try:
                            asof = date.fromisoformat(run_asof)
                            if asof not in run_dirs_by_date:
                                run_dirs_by_date[asof] = run_dir
                        except ValueError:
                            pass

    records = []

    for asof_date in common_dates:
        features_df = read_pair_state_features(config, feature_version, asof_date)
        if features_df is None:
            continue

        # Get decisions for this date if available
        run_dir = run_dirs_by_date.get(asof_date)
        decisions_df = None
        if run_dir:
            decisions_path = run_dir / "decisions.parquet"
            if decisions_path.exists():
                decisions_df = pl.read_parquet(decisions_path)

        for feat_row in features_df.iter_rows(named=True):
            pair_id = feat_row["pair_id"]

            # Get action from decisions if available
            action_units = 0.0
            ticket_id = None
            conviction = 3
            max_gross = 1000000.0
            orientation = "LONG_SPREAD"
            horizon_days = 30
            time_stop_days = 20

            if decisions_df is not None:
                dec_row = decisions_df.filter(pl.col("pair_id") == pair_id)
                if not dec_row.is_empty():
                    dec = dec_row.row(0, named=True)
                    action_units = dec.get("action_units", 0.0)
                    ticket_id = dec.get("ticket_id")

            # Compute 5-day forward reward
            reward_5d = get_future_pnl_for_pair(
                config, label_version, pair_id, asof_date, reward_horizon
            )

            if reward_5d is None:
                continue

            records.append({
                "asof_date": asof_date,
                "pair_id": pair_id,
                "ticket_id": ticket_id or f"{pair_id}_unknown",
                # State features
                "zscore": feat_row.get("zscore", 0.0),
                "beta": feat_row.get("beta", 1.0),
                "beta_stability": feat_row.get("beta_stability", 0.0),
                "spread": feat_row.get("spread", 0.0),
                "no_mean_cross_days": feat_row.get("no_mean_cross_days", 0),
                "extreme_z": feat_row.get("extreme_z", 0),
                # Ticket conditioning
                "conviction_scaled": _scale_conviction(conviction),
                "max_gross_notional_scaled": _scale_notional(max_gross),
                "orientation_long_spread": 1 if orientation == "LONG_SPREAD" else 0,
                "orientation_short_spread": 1 if orientation == "SHORT_SPREAD" else 0,
                "horizon_scaled": _scale_horizon(horizon_days),
                "time_stop_scaled": _scale_time_stop(time_stop_days),
                # Action and reward
                "action_units": action_units,
                "reward_5d": reward_5d,
                # Metadata
                "feature_version": feature_version,
                "label_version": label_version,
            })

    if not records:
        return _empty_training_dataframe()

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
    ])
    df = df.sort(["asof_date", "pair_id"])

    return df


def _empty_training_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "asof_date": pl.Date,
            "pair_id": pl.Utf8,
            "ticket_id": pl.Utf8,
            "zscore": pl.Float64,
            "beta": pl.Float64,
            "beta_stability": pl.Float64,
            "spread": pl.Float64,
            "no_mean_cross_days": pl.Int64,
            "extreme_z": pl.Int8,
            "conviction_scaled": pl.Float64,
            "max_gross_notional_scaled": pl.Float64,
            "orientation_long_spread": pl.Int8,
            "orientation_short_spread": pl.Int8,
            "horizon_scaled": pl.Float64,
            "time_stop_scaled": pl.Float64,
            "action_units": pl.Float64,
            "reward_5d": pl.Float64,
            "feature_version": pl.Utf8,
            "label_version": pl.Utf8,
        }
    )


def _snap_to_allowed_action(value: float) -> float:
    # Snap to nearest allowed action
    best_action = 0.0
    best_dist = float("inf")
    for action in ALLOWED_ACTIONS:
        dist = abs(value - action)
        if dist < best_dist:
            best_dist = dist
            best_action = action
    return best_action


def _scale_conviction(conviction: int) -> float:
    # Scale conviction from [1, 5] to [0, 1]
    return (conviction - 1) / 4.0


def _scale_notional(notional: float) -> float:
    # Log-scale notional and normalize
    # Assume typical range is 100k to 10M
    import math
    if notional <= 0:
        return 0.0
    log_val = math.log10(notional)
    # Map log10(100000)=5 to 0, log10(10000000)=7 to 1
    scaled = (log_val - 5.0) / 2.0
    return max(0.0, min(1.0, scaled))


def _scale_horizon(horizon_days: int) -> float:
    # Scale horizon from [1, 90] to [0, 1]
    return min(1.0, max(0.0, (horizon_days - 1) / 89.0))


def _scale_time_stop(time_stop_days: int) -> float:
    # Scale time stop from [1, 60] to [0, 1]
    return min(1.0, max(0.0, (time_stop_days - 1) / 59.0))


def get_state_feature_columns() -> List[str]:
    return [
        "zscore",
        "beta",
        "beta_stability",
        "spread",
        "no_mean_cross_days",
        "extreme_z",
        "conviction_scaled",
        "max_gross_notional_scaled",
        "orientation_long_spread",
        "orientation_short_spread",
        "horizon_scaled",
        "time_stop_scaled",
    ]
