# RL Controller - uses trained contextual bandit policy for action selection

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

from src.tickets import PairTicket
from src.config import Config
from src.training_dataset import get_state_feature_columns, ALLOWED_ACTIONS


# Global policy cache
_policy_cache: Dict[str, Any] = {}


def _get_policy(config: Config, model_version: str):
    # Lazy load policy to avoid circular imports
    cache_key = f"{config.pair_policy_dir}_{model_version}"

    if cache_key not in _policy_cache:
        from src.train import load_policy
        policy = load_policy(config, model_version)
        _policy_cache[cache_key] = policy

    return _policy_cache[cache_key]


def compute_target_units(
    ticket: PairTicket,
    zscore: float,
    seed: int = 42,
    config: Optional[Config] = None,
    model_version: Optional[str] = None,
    beta: float = 1.0,
    beta_stability: float = 0.0,
    spread: float = 0.0,
    no_mean_cross_days: int = 0,
    extreme_z: int = 0,
) -> float:
    # Compute target units using trained policy if available
    # Falls back to zero (no position) if no model is loaded

    # If no config or model_version provided, return 0 (stub behavior)
    if config is None or model_version is None:
        return 0.0

    # Try to load the policy
    policy = _get_policy(config, model_version)

    if policy is None or not policy.is_fitted:
        # No trained model available - return no position
        return 0.0

    # Construct state features
    # Scale ticket conditioning features
    conviction_scaled = (ticket.conviction - 1) / 4.0
    max_gross_scaled = _scale_notional(ticket.max_gross_notional)
    orientation_long = 1 if ticket.orientation == "LONG_SPREAD" else 0
    orientation_short = 1 if ticket.orientation == "SHORT_SPREAD" else 0
    horizon_scaled = min(1.0, max(0.0, (ticket.horizon_days - 1) / 89.0))
    time_stop_scaled = min(1.0, max(0.0, (ticket.time_stop_days - 1) / 59.0))

    # Build feature vector in the same order as training
    state_features = np.array([
        zscore,
        beta,
        beta_stability,
        spread,
        no_mean_cross_days,
        extreme_z,
        conviction_scaled,
        max_gross_scaled,
        orientation_long,
        orientation_short,
        horizon_scaled,
        time_stop_scaled,
    ])

    # Select action using policy (greedy for determinism)
    action = policy.select_action(state_features, greedy=True)

    return action


def compute_notionals(
    units: float,
    beta: float,
    max_gross_notional: float,
    ticket: PairTicket,
) -> Tuple[float, float]:
    # Same notional computation as baseline
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


def get_controller_version(model_version: Optional[str] = None) -> str:
    if model_version:
        return f"rl_bandit_{model_version}"
    return "rl_stub_v1"


def _scale_notional(notional: float) -> float:
    # Log-scale notional and normalize
    import math
    if notional <= 0:
        return 0.0
    log_val = math.log10(notional)
    scaled = (log_val - 5.0) / 2.0
    return max(0.0, min(1.0, scaled))


def clear_policy_cache() -> None:
    # Clear cached policies (useful for testing or reloading)
    global _policy_cache
    _policy_cache = {}
