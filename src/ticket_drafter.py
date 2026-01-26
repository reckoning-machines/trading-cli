# Ticket Drafter - creates and validates draft tickets

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml

from src.config import Config
from src.tickets import validate_ticket_data, PairTicket


# Required fields for a complete ticket
REQUIRED_FIELDS = [
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

# Optional fields with defaults
OPTIONAL_FIELDS = {
    "entry_z": 1.5,
    "exit_z": 0.2,
}

# Valid values for certain fields
VALID_ORIENTATIONS = ["LONG_SPREAD", "SHORT_SPREAD"]
VALID_TYPES = ["PAIR"]


def lint_ticket_yaml(yaml_content: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    # Parse and lint a YAML ticket string
    # Returns (errors, warnings, parsed_data)
    errors = []
    warnings = []
    parsed_data = {}

    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            errors.append("YAML content must be a dictionary/mapping")
            return errors, warnings, parsed_data
        parsed_data = data
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        return errors, warnings, parsed_data

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate field types and values
    if "type" in data and data["type"] not in VALID_TYPES:
        errors.append(f"Invalid type '{data['type']}'. Must be one of: {VALID_TYPES}")

    if "orientation" in data and data["orientation"] not in VALID_ORIENTATIONS:
        errors.append(f"Invalid orientation '{data['orientation']}'. Must be one of: {VALID_ORIENTATIONS}")

    if "conviction" in data:
        try:
            conv = int(data["conviction"])
            if not 1 <= conv <= 5:
                errors.append(f"Conviction must be 1-5, got: {conv}")
        except (TypeError, ValueError):
            errors.append(f"Conviction must be an integer, got: {data['conviction']}")

    if "max_gross_notional" in data:
        try:
            val = float(data["max_gross_notional"])
            if val <= 0:
                errors.append("max_gross_notional must be positive")
        except (TypeError, ValueError):
            errors.append(f"max_gross_notional must be a number, got: {data['max_gross_notional']}")

    if "max_leg_notional" in data:
        try:
            val = float(data["max_leg_notional"])
            if val <= 0:
                errors.append("max_leg_notional must be positive")
        except (TypeError, ValueError):
            errors.append(f"max_leg_notional must be a number, got: {data['max_leg_notional']}")

    if "horizon_days" in data:
        try:
            val = int(data["horizon_days"])
            if val <= 0:
                errors.append("horizon_days must be positive")
        except (TypeError, ValueError):
            errors.append(f"horizon_days must be an integer, got: {data['horizon_days']}")

    if "time_stop_days" in data:
        try:
            val = int(data["time_stop_days"])
            if val <= 0:
                errors.append("time_stop_days must be positive")
        except (TypeError, ValueError):
            errors.append(f"time_stop_days must be an integer, got: {data['time_stop_days']}")

    if "expires_on" in data:
        try:
            if isinstance(data["expires_on"], date):
                pass  # Already a date
            else:
                date.fromisoformat(str(data["expires_on"]))
        except (TypeError, ValueError):
            errors.append(f"expires_on must be a valid date (YYYY-MM-DD), got: {data['expires_on']}")

    if "entry_z" in data:
        try:
            val = float(data["entry_z"])
            if val <= 0:
                warnings.append("entry_z is typically positive (e.g., 1.5)")
        except (TypeError, ValueError):
            errors.append(f"entry_z must be a number, got: {data['entry_z']}")

    if "exit_z" in data:
        try:
            val = float(data["exit_z"])
            if val < 0:
                warnings.append("exit_z is typically non-negative (e.g., 0.2)")
        except (TypeError, ValueError):
            errors.append(f"exit_z must be a number, got: {data['exit_z']}")

    # Cross-field validations
    if "max_gross_notional" in data and "max_leg_notional" in data:
        try:
            gross = float(data["max_gross_notional"])
            leg = float(data["max_leg_notional"])
            if leg > gross:
                warnings.append("max_leg_notional is greater than max_gross_notional")
        except (TypeError, ValueError):
            pass

    if "entry_z" in data and "exit_z" in data:
        try:
            entry = float(data["entry_z"])
            exit_z = float(data["exit_z"])
            if exit_z >= entry:
                warnings.append("exit_z should be less than entry_z for mean reversion")
        except (TypeError, ValueError):
            pass

    # Normalize symbols
    if "leg_a" in data:
        parsed_data["leg_a"] = str(data["leg_a"]).upper()
    if "leg_b" in data:
        parsed_data["leg_b"] = str(data["leg_b"]).upper()

    return errors, warnings, parsed_data


def generate_draft_ticket_id(leg_a: str, leg_b: str) -> str:
    # Generate a draft ticket ID
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{leg_a}_{leg_b}_draft_{ts}"


def create_draft_from_params(
    config: Config,
    leg_a: str,
    leg_b: str,
    orientation: str,
    max_gross_notional: float,
    max_leg_notional: float,
    conviction: int,
    horizon_days: int,
    time_stop_days: int,
    expires_on: date,
    flatten_on_breakdown: bool = True,
    entry_z: Optional[float] = None,
    exit_z: Optional[float] = None,
    ticket_id: Optional[str] = None,
) -> Tuple[Path, str]:
    # Create a draft ticket file from parameters
    # Returns (draft_path, yaml_content)

    config.tickets_drafts_dir.mkdir(parents=True, exist_ok=True)

    leg_a = leg_a.upper()
    leg_b = leg_b.upper()

    if ticket_id is None:
        ticket_id = generate_draft_ticket_id(leg_a, leg_b)

    ticket_data = {
        "ticket_id": ticket_id,
        "type": "PAIR",
        "leg_a": leg_a,
        "leg_b": leg_b,
        "orientation": orientation,
        "max_gross_notional": max_gross_notional,
        "max_leg_notional": max_leg_notional,
        "horizon_days": horizon_days,
        "conviction": conviction,
        "time_stop_days": time_stop_days,
        "flatten_on_breakdown": flatten_on_breakdown,
        "expires_on": expires_on.isoformat() if isinstance(expires_on, date) else expires_on,
    }

    if entry_z is not None:
        ticket_data["entry_z"] = entry_z
    if exit_z is not None:
        ticket_data["exit_z"] = exit_z

    yaml_content = yaml.dump(ticket_data, default_flow_style=False, sort_keys=False)

    draft_filename = f"{ticket_id}.yaml"
    draft_path = config.tickets_drafts_dir / draft_filename

    with open(draft_path, "w") as f:
        f.write(yaml_content)

    return draft_path, yaml_content


def save_draft_yaml(config: Config, yaml_content: str, filename: Optional[str] = None) -> Path:
    # Save raw YAML content as a draft file
    config.tickets_drafts_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"draft_{ts}.yaml"

    if not filename.endswith(".yaml") and not filename.endswith(".yml"):
        filename = f"{filename}.yaml"

    draft_path = config.tickets_drafts_dir / filename

    with open(draft_path, "w") as f:
        f.write(yaml_content)

    return draft_path


def suggest_improvements(parsed_data: Dict[str, Any]) -> List[str]:
    # Suggest improvements to a ticket
    suggestions = []

    # Check conviction vs notional sizing
    if "conviction" in parsed_data and "max_gross_notional" in parsed_data:
        conv = int(parsed_data["conviction"])
        gross = float(parsed_data["max_gross_notional"])

        if conv <= 2 and gross > 500000:
            suggestions.append(
                f"Low conviction ({conv}) but high max_gross_notional ({gross:,.0f}). "
                "Consider reducing notional for low-conviction trades."
            )

        if conv >= 4 and gross < 100000:
            suggestions.append(
                f"High conviction ({conv}) but low max_gross_notional ({gross:,.0f}). "
                "Consider increasing notional for high-conviction trades."
            )

    # Check time stop vs horizon
    if "time_stop_days" in parsed_data and "horizon_days" in parsed_data:
        time_stop = int(parsed_data["time_stop_days"])
        horizon = int(parsed_data["horizon_days"])

        if time_stop > horizon:
            suggestions.append(
                f"time_stop_days ({time_stop}) > horizon_days ({horizon}). "
                "Consider setting time_stop <= horizon for consistent exit timing."
            )

    # Check entry/exit z-score
    entry_z = parsed_data.get("entry_z", 1.5)
    exit_z = parsed_data.get("exit_z", 0.2)

    if entry_z < 1.0:
        suggestions.append(
            f"entry_z ({entry_z}) is low. Typical values are 1.5-2.5 for mean reversion."
        )

    if exit_z > 0.5:
        suggestions.append(
            f"exit_z ({exit_z}) is high. Typical values are 0.1-0.3 to capture full reversion."
        )

    return suggestions


def get_missing_fields(parsed_data: Dict[str, Any]) -> List[str]:
    # Return list of required fields not present in parsed_data
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in parsed_data:
            missing.append(field)
    return missing


def get_field_questions() -> Dict[str, str]:
    # Return questions to ask for each field
    return {
        "leg_a": "What is the first leg symbol (e.g., JPM)?",
        "leg_b": "What is the second leg symbol (e.g., BAC)?",
        "orientation": "What is the orientation? (LONG_SPREAD or SHORT_SPREAD)",
        "max_gross_notional": "What is the maximum gross notional in dollars?",
        "max_leg_notional": "What is the maximum per-leg notional in dollars?",
        "conviction": "What is your conviction level? (1-5, where 5 is highest)",
        "horizon_days": "What is the expected horizon in days?",
        "time_stop_days": "After how many days should we exit regardless of signal?",
        "expires_on": "When should this ticket expire? (YYYY-MM-DD)",
        "flatten_on_breakdown": "Should we flatten on breakdown (extreme z-score)? (yes/no)",
        "entry_z": "What z-score threshold for entry? (default: 1.5)",
        "exit_z": "What z-score threshold for exit? (default: 0.2)",
    }
