# RL Controller Stub - placeholder for reinforcement learning controller
# This module provides the same interface as controller_baseline but is a no-op stub.
# In a future version, this can be replaced with an actual RL policy.

from typing import Tuple

from src.tickets import PairTicket


ALLOWED_UNITS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def compute_target_units(
    ticket: PairTicket,
    zscore: float,
    seed: int = 42,
) -> float:
    # Fix 3: Orientation is sourced from ticket only (ticket.orientation)
    # RL stub: returns 0 (no position) for all inputs
    # In a real implementation, this would:
    # 1. Load a trained policy model
    # 2. Construct the state representation from features
    # 3. Run inference to get action probabilities
    # 4. Sample or argmax to get the discrete action
    _ = seed  # Seed would be used for stochastic policies
    _ = ticket.orientation  # Would be used in state construction
    return 0.0


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
    return "rl_stub_v1"
