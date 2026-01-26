# Feature computation for pair state

from datetime import date
from typing import List, Optional

import numpy as np
import polars as pl

from src.config import Config
from src.storage import read_daily_bars


def compute_rolling_beta(
    prices_a: pl.Series,
    prices_b: pl.Series,
    window: int,
) -> pl.Series:
    # Fix 1: Use log returns for beta computation to match spread definition
    log_a = prices_a.log()
    log_b = prices_b.log()
    returns_a = log_a.diff()
    returns_b = log_b.diff()

    betas = []
    for i in range(len(prices_a)):
        if i < window:
            betas.append(None)
        else:
            ret_a = returns_a[i - window + 1 : i + 1].to_numpy()
            ret_b = returns_b[i - window + 1 : i + 1].to_numpy()

            mask = ~(np.isnan(ret_a) | np.isnan(ret_b))
            ret_a = ret_a[mask]
            ret_b = ret_b[mask]

            if len(ret_a) < 10:
                betas.append(None)
            else:
                cov = np.cov(ret_a, ret_b)
                if cov[1, 1] > 0:
                    beta = cov[0, 1] / cov[1, 1]
                    betas.append(beta)
                else:
                    betas.append(None)

    return pl.Series(betas)


def compute_spread(
    prices_a: pl.Series,
    prices_b: pl.Series,
    beta: float,
) -> pl.Series:
    log_a = prices_a.log()
    log_b = prices_b.log()
    spread = log_a - beta * log_b
    return spread


def compute_zscore(spread: pl.Series, window: int) -> pl.Series:
    mean = spread.rolling_mean(window_size=window)
    std = spread.rolling_std(window_size=window)
    zscore = (spread - mean) / std
    return zscore


def compute_beta_stability(betas: pl.Series, window: int) -> pl.Series:
    return betas.rolling_std(window_size=window)


def compute_no_mean_cross_days(zscore: pl.Series) -> int:
    if zscore.is_empty() or zscore.is_null().all():
        return 0

    zscore_list = zscore.to_list()
    days = 0

    for i in range(len(zscore_list) - 1, 0, -1):
        current = zscore_list[i]
        previous = zscore_list[i - 1]

        if current is None or previous is None:
            break

        if (current >= 0 and previous < 0) or (current <= 0 and previous > 0):
            break

        days += 1

    return days


def compute_pair_features(
    config: Config,
    leg_a: str,
    leg_b: str,
    asof_date: date,
    window_hedge: int,
    window_z: int,
    feature_version: str,
) -> Optional[dict]:
    prices_a_df = read_daily_bars(config, leg_a)
    prices_b_df = read_daily_bars(config, leg_b)

    if prices_a_df is None or prices_b_df is None:
        return None

    prices_a_df = prices_a_df.filter(pl.col("date") <= asof_date).sort("date")
    prices_b_df = prices_b_df.filter(pl.col("date") <= asof_date).sort("date")

    if prices_a_df.is_empty() or prices_b_df.is_empty():
        return None

    merged = prices_a_df.select([
        pl.col("date"),
        pl.col("adj_close").alias("price_a"),
    ]).join(
        prices_b_df.select([
            pl.col("date"),
            pl.col("adj_close").alias("price_b"),
        ]),
        on="date",
        how="inner",
    ).sort("date")

    if merged.height < window_hedge + window_z:
        return None

    prices_a = merged["price_a"]
    prices_b = merged["price_b"]

    betas = compute_rolling_beta(prices_a, prices_b, window_hedge)
    merged = merged.with_columns([betas.alias("beta")])

    # Fix 1: Get beta from asof_date row, not last non-null
    last_row = merged.filter(pl.col("date") == asof_date)
    if last_row.is_empty():
        last_row = merged.tail(1)

    if last_row.is_empty():
        return None

    last_beta = last_row["beta"][0]

    if last_beta is None or not np.isfinite(last_beta):
        return None

    # Compute spread and zscore using the asof_date beta
    spread = compute_spread(prices_a, prices_b, last_beta)
    merged = merged.with_columns([spread.alias("spread")])

    zscore = compute_zscore(spread, window_z)
    merged = merged.with_columns([zscore.alias("zscore")])

    beta_stability = compute_beta_stability(betas, window_hedge)
    merged = merged.with_columns([beta_stability.alias("beta_stability")])

    # Re-fetch last_row after adding computed columns
    last_row = merged.filter(pl.col("date") == asof_date)
    if last_row.is_empty():
        last_row = merged.tail(1)

    if last_row.is_empty():
        return None

    last_spread = last_row["spread"][0]
    last_zscore = last_row["zscore"][0]
    last_beta_stability = last_row["beta_stability"][0]

    if last_zscore is None or not np.isfinite(last_zscore):
        return None

    no_mean_cross_days = compute_no_mean_cross_days(zscore)
    extreme_z = 1 if abs(last_zscore) >= 3.0 else 0

    return {
        "asof_date": asof_date,
        "pair_id": f"{leg_a}__{leg_b}",
        "leg_a": leg_a,
        "leg_b": leg_b,
        "window_hedge": window_hedge,
        "window_z": window_z,
        "beta": float(last_beta),
        "spread": float(last_spread) if last_spread is not None else 0.0,
        "zscore": float(last_zscore),
        "beta_stability": float(last_beta_stability) if last_beta_stability is not None else 0.0,
        "no_mean_cross_days": no_mean_cross_days,
        "extreme_z": extreme_z,
        "feature_version": feature_version,
    }


def build_pair_state_dataframe(features_list: List[dict]) -> pl.DataFrame:
    if not features_list:
        return pl.DataFrame(
            schema={
                "asof_date": pl.Date,
                "pair_id": pl.Utf8,
                "leg_a": pl.Utf8,
                "leg_b": pl.Utf8,
                "window_hedge": pl.Int64,
                "window_z": pl.Int64,
                "beta": pl.Float64,
                "spread": pl.Float64,
                "zscore": pl.Float64,
                "beta_stability": pl.Float64,
                "no_mean_cross_days": pl.Int64,
                "extreme_z": pl.Int8,
                "feature_version": pl.Utf8,
            }
        )

    df = pl.DataFrame(features_list)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
        pl.col("pair_id").cast(pl.Utf8),
        pl.col("leg_a").cast(pl.Utf8),
        pl.col("leg_b").cast(pl.Utf8),
        pl.col("window_hedge").cast(pl.Int64),
        pl.col("window_z").cast(pl.Int64),
        pl.col("beta").cast(pl.Float64),
        pl.col("spread").cast(pl.Float64),
        pl.col("zscore").cast(pl.Float64),
        pl.col("beta_stability").cast(pl.Float64),
        pl.col("no_mean_cross_days").cast(pl.Int64),
        pl.col("extreme_z").cast(pl.Int8),
        pl.col("feature_version").cast(pl.Utf8),
    ])

    return df


def compute_spread_volatility(
    config: Config,
    leg_a: str,
    leg_b: str,
    beta: float,
    asof_date: date,
    lookback: int = 20,
) -> float:
    prices_a_df = read_daily_bars(config, leg_a)
    prices_b_df = read_daily_bars(config, leg_b)

    if prices_a_df is None or prices_b_df is None:
        return 0.02

    prices_a_df = prices_a_df.filter(pl.col("date") <= asof_date).sort("date").tail(lookback + 1)
    prices_b_df = prices_b_df.filter(pl.col("date") <= asof_date).sort("date").tail(lookback + 1)

    merged = prices_a_df.select([
        pl.col("date"),
        pl.col("adj_close").alias("price_a"),
    ]).join(
        prices_b_df.select([
            pl.col("date"),
            pl.col("adj_close").alias("price_b"),
        ]),
        on="date",
        how="inner",
    ).sort("date")

    if merged.height < 2:
        return 0.02

    spread = merged["price_a"].log() - beta * merged["price_b"].log()
    spread_returns = spread.diff().drop_nulls()

    if spread_returns.is_empty():
        return 0.02

    vol = float(spread_returns.std())
    return vol if np.isfinite(vol) else 0.02
