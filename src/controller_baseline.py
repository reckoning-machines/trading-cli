# Baseline Controller - deterministic mean-reversion policy

from typing import Tuple

from src.tickets import PairTicket


ALLOWED_UNITS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def compute_target_units(
    ticket: PairTicket,
    zscore: float,
) -> float:
    # Fix 3: Orientation is sourced from ticket only, no separate parameter
    entry_z = ticket.get_entry_z()
    exit_z = ticket.get_exit_z()
    orientation = ticket.orientation

    abs_z = abs(zscore)

    if abs_z < exit_z:
        return 0.0

    if abs_z < entry_z:
        return 0.0

    if zscore > 0:
        raw_units = -1.0
    else:
        raw_units = 1.0

    if orientation == "SHORT_SPREAD":
        raw_units = -raw_units

    max_units = get_max_units_by_conviction(ticket.conviction)
    scaled_units = raw_units * max_units

    return clamp_to_allowed(scaled_units)


def get_max_units_by_conviction(conviction: int) -> float:
    if conviction <= 2:
        return 0.5
    return 1.0


def clamp_to_allowed(units: float) -> float:
    if units == 0:
        return 0.0
    closest = min(ALLOWED_UNITS, key=lambda x: abs(x - units))
    return closest


def compute_notionals(
    units: float,
    beta: float,
    max_gross_notional: float,
    ticket: PairTicket,
) -> Tuple[float, float]:
    # Fix 3: Orientation is sourced from ticket only
    if units == 0:
        return 0.0, 0.0

    orientation = ticket.orientation

    abs_units = abs(units)
    target_gross = abs_units * max_gross_notional

    denom = 1.0 + abs(beta)
    if denom == 0:
        denom = 1.0

    N = target_gross / denom

    sign_unit = 1.0 if units > 0 else -1.0

    notional_a = sign_unit * N

    if orientation == "LONG_SPREAD":
        notional_b = -sign_unit * abs(beta) * N
    else:
        notional_b = sign_unit * abs(beta) * N

    return notional_a, notional_b


def get_controller_version() -> str:
    return "baseline_v1"
