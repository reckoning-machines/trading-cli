# Universe management - defines tradeable symbols

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from src.config import Config


DEFAULT_TICKERS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "PNC"]
DEFAULT_CONTEXT_ONLY = ["XLF", "KRE"]


@dataclass
class PairSuggestion:
    leg_a: str
    leg_b: str

    @property
    def pair_id(self) -> str:
        return f"{self.leg_a}__{self.leg_b}"


@dataclass
class Universe:
    tickers: List[str] = field(default_factory=list)
    context_only: List[str] = field(default_factory=list)
    pair_suggestions: List[PairSuggestion] = field(default_factory=list)

    @property
    def all_symbols(self) -> List[str]:
        return sorted(set(self.tickers + self.context_only))

    @property
    def tradeable_symbols(self) -> List[str]:
        return sorted(set(self.tickers))


def load_universe(config: Config) -> Universe:
    universe_path = Path(config.universe_file)

    if not universe_path.exists():
        return Universe(
            tickers=DEFAULT_TICKERS.copy(),
            context_only=DEFAULT_CONTEXT_ONLY.copy(),
            pair_suggestions=[],
        )

    with open(universe_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return Universe(
            tickers=DEFAULT_TICKERS.copy(),
            context_only=DEFAULT_CONTEXT_ONLY.copy(),
            pair_suggestions=[],
        )

    tickers = [str(t).upper() for t in data.get("tickers", DEFAULT_TICKERS)]
    context_only = [str(t).upper() for t in data.get("context_only", DEFAULT_CONTEXT_ONLY)]

    pair_suggestions = []
    for pair_data in data.get("pair_suggestions", []):
        pair_suggestions.append(PairSuggestion(
            leg_a=str(pair_data["leg_a"]).upper(),
            leg_b=str(pair_data["leg_b"]).upper(),
        ))

    return Universe(
        tickers=tickers,
        context_only=context_only,
        pair_suggestions=pair_suggestions,
    )
