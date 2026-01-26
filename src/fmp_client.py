# FMP (Financial Modeling Prep) API client

import hashlib
from datetime import date, datetime
from typing import Optional

import polars as pl
import requests

from src.config import Config


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPClient:
    def __init__(self, config: Config):
        self.api_key = config.fmp_api_key
        self.config = config

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
        params = {
            "apikey": self.api_key,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "historical" not in data or not data["historical"]:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.Utf8,
                    "adj_close": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        records = []
        for row in data["historical"]:
            records.append({
                "date": datetime.strptime(row["date"], "%Y-%m-%d").date(),
                "symbol": symbol.upper(),
                "adj_close": float(row.get("adjClose", row.get("close", 0))),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })

        df = pl.DataFrame(records)
        df = df.with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("adj_close").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ])

        return df.sort("date")


def compute_data_checksum(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "empty"
    content = df.write_csv()
    return hashlib.sha256(content.encode()).hexdigest()[:16]
