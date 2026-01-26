# Risk Governor - applies position limits and clamps

from dataclasses import dataclass
from typing import List

from src.tickets import PairTicket


CLAMP_GROSS = "GROSS_CLAMP"
CLAMP_LEG_A = "LEG_CLAMP_A"
CLAMP_LEG_B = "LEG_CLAMP_B"
CLAMP_BREAKDOWN = "BREAKDOWN_FLATTEN"
CLAMP_EXPIRED = "EXPIRED_SKIP"


@dataclass
class GovernorResult:
    # Fix 2: Separate action_units (controller discrete output) from executed_units (post-scale)
    action_units: float
    executed_units: float
    executed_notional_a: float
    executed_notional_b: float
    clamp_codes: List[str]


def apply_risk_governor(
    ticket: PairTicket,
    beta: float,
    proposed_units: float,
    proposed_notional_a: float,
    proposed_notional_b: float,
    extreme_z: int,
    is_expired: bool,
) -> GovernorResult:
    # Fix 2: action_units is the discrete controller output (proposed_units)
    # Fix 2: executed_units is the continuous post-scale value
    # Fix 2: No discretization after scaling - notionals and executed_units remain consistent
    clamp_codes = []

    if is_expired:
        return GovernorResult(
            action_units=proposed_units,
            executed_units=0.0,
            executed_notional_a=0.0,
            executed_notional_b=0.0,
            clamp_codes=[CLAMP_EXPIRED],
        )

    if ticket.flatten_on_breakdown and extreme_z == 1:
        return GovernorResult(
            action_units=proposed_units,
            executed_units=0.0,
            executed_notional_a=0.0,
            executed_notional_b=0.0,
            clamp_codes=[CLAMP_BREAKDOWN],
        )

    executed_notional_a = proposed_notional_a
    executed_notional_b = proposed_notional_b
    executed_units = proposed_units

    if abs(executed_notional_a) > ticket.max_leg_notional:
        scale = ticket.max_leg_notional / abs(executed_notional_a)
        executed_notional_a = executed_notional_a * scale
        executed_notional_b = executed_notional_b * scale
        executed_units = executed_units * scale
        clamp_codes.append(CLAMP_LEG_A)

    if abs(executed_notional_b) > ticket.max_leg_notional:
        scale = ticket.max_leg_notional / abs(executed_notional_b)
        executed_notional_a = executed_notional_a * scale
        executed_notional_b = executed_notional_b * scale
        executed_units = executed_units * scale
        clamp_codes.append(CLAMP_LEG_B)

    gross_notional = abs(executed_notional_a) + abs(executed_notional_b)
    if gross_notional > ticket.max_gross_notional:
        scale = ticket.max_gross_notional / gross_notional
        executed_notional_a = executed_notional_a * scale
        executed_notional_b = executed_notional_b * scale
        executed_units = executed_units * scale
        clamp_codes.append(CLAMP_GROSS)

    # Fix 2: Sort clamp_codes for deterministic output
    clamp_codes = sorted(clamp_codes)

    return GovernorResult(
        action_units=proposed_units,
        executed_units=executed_units,
        executed_notional_a=executed_notional_a,
        executed_notional_b=executed_notional_b,
        clamp_codes=clamp_codes,
    )
