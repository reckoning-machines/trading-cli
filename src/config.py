# Configuration module - loads settings from environment variables

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    fmp_api_key: str
    data_dir: Path
    runs_dir: Path
    tickets_dir: Path
    feature_version: str
    universe_file: Path

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.runs_dir = Path(self.runs_dir)
        self.tickets_dir = Path(self.tickets_dir)
        self.universe_file = Path(self.universe_file)

    @property
    def fmp_cache_dir(self) -> Path:
        return self.data_dir / "fmp_cache"

    @property
    def daily_bars_dir(self) -> Path:
        return self.fmp_cache_dir / "daily_bars"

    @property
    def fmp_meta_dir(self) -> Path:
        return self.fmp_cache_dir / "meta"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def pair_state_dir(self) -> Path:
        return self.features_dir / "pair_state"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def realized_labels_dir(self) -> Path:
        return self.labels_dir / "realized"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def last_positions_dir(self) -> Path:
        return self.state_dir / "last_positions"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def pair_policy_dir(self) -> Path:
        return self.models_dir / "pair_policy"


def load_config() -> Config:
    fmp_api_key = os.environ.get("FMP_API_KEY")
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY environment variable is required")

    return Config(
        fmp_api_key=fmp_api_key,
        data_dir=os.environ.get("DATA_DIR", "./data"),
        runs_dir=os.environ.get("RUNS_DIR", "./runs"),
        tickets_dir=os.environ.get("TICKETS_DIR", "./tickets/active"),
        feature_version=os.environ.get("FEATURE_VERSION", "v1"),
        universe_file=os.environ.get("UNIVERSE_FILE", "./universe.yaml"),
    )


def get_config_or_exit() -> Config:
    try:
        return load_config()
    except ValueError as e:
        raise SystemExit(f"Configuration error: {e}")
