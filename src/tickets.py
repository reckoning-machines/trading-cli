# Ticket management - human-in-the-loop trading tickets

import hashlib
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import yaml

from src.config import Config


@dataclass
class PairTicket:
    ticket_id: str
    ticket_type: str
    leg_a: str
    leg_b: str
    orientation: str
    max_gross_notional: float
    max_leg_notional: float
    horizon_days: int
    conviction: int
    time_stop_days: int
    flatten_on_breakdown: bool
    expires_on: date
    entry_z: Optional[float] = None
    exit_z: Optional[float] = None
    source_file: Optional[str] = None

    @property
    def pair_id(self) -> str:
        return f"{self.leg_a}__{self.leg_b}"

    def is_expired(self, asof_date: date) -> bool:
        return self.expires_on < asof_date

    def get_entry_z(self) -> float:
        return self.entry_z if self.entry_z is not None else 1.5

    def get_exit_z(self) -> float:
        return self.exit_z if self.exit_z is not None else 0.2


def parse_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Cannot parse date from: {value}")


def validate_ticket_data(data: dict, file_path: str) -> None:
    required_fields = [
        "ticket_id",
        "type",
        "leg_a",
        "leg_b",
        "orientation",
        "max_gross_notional",
        "max_leg_notional",
        "horizon_days",
        "conviction",
        "time_stop_days",
        "flatten_on_breakdown",
        "expires_on",
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in ticket file: {file_path}")

    if data["type"] != "PAIR":
        raise ValueError(f"Invalid ticket type '{data['type']}' in {file_path}. Must be PAIR.")

    if data["orientation"] not in ("LONG_SPREAD", "SHORT_SPREAD"):
        raise ValueError(
            f"Invalid orientation '{data['orientation']}' in {file_path}. "
            "Must be LONG_SPREAD or SHORT_SPREAD."
        )

    if not 1 <= data["conviction"] <= 5:
        raise ValueError(
            f"Invalid conviction '{data['conviction']}' in {file_path}. Must be 1 to 5."
        )


def load_ticket_from_file(file_path: Path) -> PairTicket:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    validate_ticket_data(data, str(file_path))

    return PairTicket(
        ticket_id=str(data["ticket_id"]),
        ticket_type=data["type"],
        leg_a=str(data["leg_a"]).upper(),
        leg_b=str(data["leg_b"]).upper(),
        orientation=data["orientation"],
        max_gross_notional=float(data["max_gross_notional"]),
        max_leg_notional=float(data["max_leg_notional"]),
        horizon_days=int(data["horizon_days"]),
        conviction=int(data["conviction"]),
        time_stop_days=int(data["time_stop_days"]),
        flatten_on_breakdown=bool(data["flatten_on_breakdown"]),
        expires_on=parse_date(data["expires_on"]),
        entry_z=float(data["entry_z"]) if data.get("entry_z") is not None else None,
        exit_z=float(data["exit_z"]) if data.get("exit_z") is not None else None,
        source_file=str(file_path),
    )


def compute_file_hash(file_path: Path) -> str:
    # Fix 5: Compute SHA256 hash of file contents
    with open(file_path, "rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()


def load_active_tickets(config: Config, asof_date: date) -> List[PairTicket]:
    # Fix 4: Sort ticket files before loading for deterministic ordering
    tickets = []
    tickets_dir = Path(config.tickets_dir)

    if not tickets_dir.exists():
        return tickets

    # Collect all yaml/yml files and sort them
    yaml_files = sorted(list(tickets_dir.glob("*.yaml")))
    yml_files = sorted(list(tickets_dir.glob("*.yml")))
    all_files = sorted(yaml_files + yml_files, key=lambda p: p.name)

    for ticket_file in all_files:
        try:
            ticket = load_ticket_from_file(ticket_file)
            if not ticket.is_expired(asof_date):
                tickets.append(ticket)
        except Exception as e:
            raise ValueError(f"Error loading ticket from {ticket_file}: {e}")

    # Fix 4: Return tickets sorted by ticket_id for deterministic ordering
    tickets.sort(key=lambda t: t.ticket_id)

    return tickets


@dataclass
class SnapshotResult:
    # Fix 5: Return both snapshot paths and hashes
    snapshot_paths: List[str]
    ticket_hashes: Dict[str, str]


def snapshot_tickets(tickets: List[PairTicket], run_dir: Path) -> SnapshotResult:
    # Fix 5: Return snapshot paths (inside run directory) and file hashes
    snapshot_dir = run_dir / "tickets_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = []
    ticket_hashes = {}

    # Fix 4: Process tickets in sorted order (by ticket_id)
    sorted_tickets = sorted(tickets, key=lambda t: t.ticket_id)

    for ticket in sorted_tickets:
        if ticket.source_file:
            src = Path(ticket.source_file)
            if src.exists():
                dst = snapshot_dir / src.name
                shutil.copy2(src, dst)
                snapshot_paths.append(str(dst))
                ticket_hashes[src.name] = compute_file_hash(src)

    return SnapshotResult(
        snapshot_paths=snapshot_paths,
        ticket_hashes=ticket_hashes,
    )


def get_required_pairs(tickets: List[PairTicket]) -> List[tuple]:
    # Fix 4: Return pairs in deterministic order
    pairs = []
    seen = set()
    # Process in sorted order by ticket_id
    for ticket in sorted(tickets, key=lambda t: t.ticket_id):
        pair_key = (ticket.leg_a, ticket.leg_b)
        if pair_key not in seen:
            pairs.append(pair_key)
            seen.add(pair_key)
    return pairs


def get_required_symbols(tickets: List[PairTicket]) -> List[str]:
    symbols = set()
    for ticket in tickets:
        symbols.add(ticket.leg_a)
        symbols.add(ticket.leg_b)
    return sorted(symbols)
