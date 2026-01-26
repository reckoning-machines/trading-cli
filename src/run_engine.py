# Run Engine - orchestrates controller runs and provides test harness

import hashlib
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import polars as pl

from src.config import Config
from src.storage import (
    ensure_directories,
    create_run_directory,
    write_manifest,
    write_decisions,
    write_blotter,
    write_metrics,
    write_logs,
    read_pair_state_features,
    get_fmp_cache_summary,
)
from src.tickets import PairTicket, load_active_tickets, snapshot_tickets
from src.features_pair_state import compute_spread_volatility
from src.risk_governor import apply_risk_governor
from src import controller_baseline
from src import controller_rl_stub


DEFAULT_COST_BPS = 1.0


def generate_run_id(asof_date: date, seed: int) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    hash_input = f"{ts}_{seed}".encode()
    short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    return f"{ts}_{short_hash}"


def get_git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def run_controller(
    config: Config,
    asof_date: date,
    feature_version: str,
    controller_name: str,
    seed: int,
) -> Dict[str, Any]:
    ensure_directories(config)

    np.random.seed(seed)

    run_id = generate_run_id(asof_date, seed)
    run_dir = create_run_directory(config, run_id)

    logs = []
    logs.append(f"Run started: {run_id}")
    logs.append(f"As-of date: {asof_date}")
    logs.append(f"Feature version: {feature_version}")
    logs.append(f"Controller: {controller_name}")
    logs.append(f"Seed: {seed}")

    tickets = load_active_tickets(config, asof_date)
    logs.append(f"Loaded {len(tickets)} active tickets")

    # Fix 5: snapshot_tickets now returns SnapshotResult with paths and hashes
    snapshot_result = snapshot_tickets(tickets, run_dir)

    features_df = read_pair_state_features(config, feature_version, asof_date)
    if features_df is None:
        logs.append("ERROR: No features found for asof_date and feature_version")
        write_logs(run_dir, "\n".join(logs))
        raise ValueError(f"No features found for {asof_date} with version {feature_version}")

    if controller_name == "baseline":
        controller_module = controller_baseline
    elif controller_name == "rl":
        controller_module = controller_rl_stub
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    controller_version = controller_module.get_controller_version()

    decisions = []
    blotter_records = []

    # Fix 4: tickets are already sorted by ticket_id from load_active_tickets
    for ticket in tickets:
        pair_id = ticket.pair_id

        pair_features = features_df.filter(pl.col("pair_id") == pair_id)
        if pair_features.is_empty():
            logs.append(f"WARNING: No features for pair {pair_id}, skipping ticket {ticket.ticket_id}")
            continue

        # Fix 6: Assert exactly one row exists for each pair
        if pair_features.height > 1:
            logs.append(f"ERROR: Multiple feature rows ({pair_features.height}) for pair {pair_id}")
            raise ValueError(f"Multiple feature rows for pair {pair_id}: expected 1, got {pair_features.height}")

        feat_row = pair_features.row(0, named=True)

        beta = feat_row["beta"]
        zscore = feat_row["zscore"]
        extreme_z = feat_row["extreme_z"]

        if not np.isfinite(beta) or not np.isfinite(zscore):
            logs.append(f"WARNING: Non-finite beta or zscore for {pair_id}, skipping")
            continue

        # Fix 3: compute_target_units takes ticket only (orientation from ticket)
        proposed_units = controller_module.compute_target_units(
            ticket=ticket,
            zscore=zscore,
        )

        # Fix 3: compute_notionals takes ticket (orientation from ticket)
        proposed_notional_a, proposed_notional_b = controller_module.compute_notionals(
            units=proposed_units,
            beta=beta,
            max_gross_notional=ticket.max_gross_notional,
            ticket=ticket,
        )

        is_expired = ticket.is_expired(asof_date)

        governor_result = apply_risk_governor(
            ticket=ticket,
            beta=beta,
            proposed_units=proposed_units,
            proposed_notional_a=proposed_notional_a,
            proposed_notional_b=proposed_notional_b,
            extreme_z=extreme_z,
            is_expired=is_expired,
        )

        # Fix 2: Use new governor result fields
        action_units = governor_result.action_units
        executed_units = governor_result.executed_units
        executed_notional_a = governor_result.executed_notional_a
        executed_notional_b = governor_result.executed_notional_b
        clamp_codes = governor_result.clamp_codes

        # Fix 2: Compute metrics using executed notionals
        gross_notional = abs(executed_notional_a) + abs(executed_notional_b)
        est_costs = (DEFAULT_COST_BPS / 10000.0) * gross_notional

        vol_proxy = compute_spread_volatility(
            config=config,
            leg_a=ticket.leg_a,
            leg_b=ticket.leg_b,
            beta=beta,
            asof_date=asof_date,
        )
        # Fix 2: est_daily_vol_dollars uses executed gross notional
        est_daily_vol_dollars = gross_notional * vol_proxy

        # Fix 2: decisions include both action_units and executed_units
        decision = {
            "run_id": run_id,
            "asof_date": asof_date,
            "ticket_id": ticket.ticket_id,
            "pair_id": pair_id,
            "action_units": action_units,
            "executed_units": executed_units,
            "clamp_codes": ",".join(clamp_codes) if clamp_codes else "",
            "executed_notional_a": executed_notional_a,
            "executed_notional_b": executed_notional_b,
            "est_costs": est_costs,
            "est_daily_vol_dollars": est_daily_vol_dollars,
            "feature_version": feature_version,
            "controller_version": controller_version,
            "zscore": zscore,
        }
        decisions.append(decision)

        # Fix 2: blotter includes both action_units and executed_units
        blotter_records.append({
            "ticket_id": ticket.ticket_id,
            "pair_id": pair_id,
            "leg_a": ticket.leg_a,
            "leg_b": ticket.leg_b,
            "action_units": action_units,
            "executed_units": executed_units,
            "executed_notional_a": executed_notional_a,
            "executed_notional_b": executed_notional_b,
            "clamp_codes": ",".join(clamp_codes) if clamp_codes else "",
        })

        logs.append(
            f"Ticket {ticket.ticket_id}: action={action_units:.2f}, "
            f"executed={executed_units:.2f}, clamps={clamp_codes}"
        )

    decisions_df = build_decisions_dataframe(decisions)
    write_decisions(run_dir, decisions_df)

    write_blotter(run_dir, blotter_records)

    metrics = compute_run_metrics(decisions)
    write_metrics(run_dir, metrics)

    # Fix 4: universe_symbols must be sorted
    # Fix 5: manifest includes ticket_hashes
    manifest = {
        "run_id": run_id,
        "asof_ts": datetime.utcnow().isoformat(),
        "asof_date": asof_date.isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "feature_version": feature_version,
        "controller_version": controller_version,
        "universe_symbols": sorted(list(set(t.leg_a for t in tickets) | set(t.leg_b for t in tickets))),
        "tickets_snapshot_paths": snapshot_result.snapshot_paths,
        "ticket_hashes": snapshot_result.ticket_hashes,
        "fmp_cache_state_summary": get_fmp_cache_summary(config),
    }
    write_manifest(run_dir, manifest)

    logs.append(f"Run completed: {run_id}")
    write_logs(run_dir, "\n".join(logs))

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "decisions": decisions,
        "metrics": metrics,
    }


def build_decisions_dataframe(decisions: List[Dict[str, Any]]) -> pl.DataFrame:
    # Fix 2: Updated schema with action_units and executed_units
    if not decisions:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "asof_date": pl.Date,
                "ticket_id": pl.Utf8,
                "pair_id": pl.Utf8,
                "action_units": pl.Float64,
                "executed_units": pl.Float64,
                "clamp_codes": pl.Utf8,
                "executed_notional_a": pl.Float64,
                "executed_notional_b": pl.Float64,
                "est_costs": pl.Float64,
                "est_daily_vol_dollars": pl.Float64,
                "feature_version": pl.Utf8,
                "controller_version": pl.Utf8,
            }
        )

    records = []
    for d in decisions:
        records.append({
            "run_id": d["run_id"],
            "asof_date": d["asof_date"],
            "ticket_id": d["ticket_id"],
            "pair_id": d["pair_id"],
            "action_units": d["action_units"],
            "executed_units": d["executed_units"],
            "clamp_codes": d["clamp_codes"],
            "executed_notional_a": d["executed_notional_a"],
            "executed_notional_b": d["executed_notional_b"],
            "est_costs": d["est_costs"],
            "est_daily_vol_dollars": d["est_daily_vol_dollars"],
            "feature_version": d["feature_version"],
            "controller_version": d["controller_version"],
        })

    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("asof_date").cast(pl.Date),
    ])

    return df


def compute_run_metrics(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not decisions:
        return {
            "total_tickets": 0,
            "total_gross_notional": 0.0,
            "total_est_costs": 0.0,
            "total_est_daily_vol": 0.0,
            "tickets_with_position": 0,
            "tickets_clamped": 0,
        }

    # Fix 2: Use executed notionals
    total_gross = sum(abs(d["executed_notional_a"]) + abs(d["executed_notional_b"]) for d in decisions)
    total_costs = sum(d["est_costs"] for d in decisions)
    total_vol = sum(d["est_daily_vol_dollars"] for d in decisions)
    with_position = sum(1 for d in decisions if d["executed_units"] != 0)
    clamped = sum(1 for d in decisions if d["clamp_codes"])

    return {
        "total_tickets": len(decisions),
        "total_gross_notional": total_gross,
        "total_est_costs": total_costs,
        "total_est_daily_vol": total_vol,
        "tickets_with_position": with_position,
        "tickets_clamped": clamped,
    }


def validate_run_artifacts(run_dir: Path) -> List[str]:
    errors = []

    required_files = [
        "manifest.json",
        "decisions.parquet",
        "blotter.csv",
        "metrics.json",
        "logs.txt",
    ]

    for fname in required_files:
        if not (run_dir / fname).exists():
            errors.append(f"Missing required file: {fname}")

    tickets_snapshot = run_dir / "tickets_snapshot"
    if not tickets_snapshot.exists():
        errors.append("Missing tickets_snapshot directory")

    return errors


def validate_decisions(decisions: List[Dict[str, Any]], tickets: List[PairTicket]) -> List[str]:
    errors = []

    for d in decisions:
        # Fix 2: Use executed notionals
        if not np.isfinite(d["executed_notional_a"]) or not np.isfinite(d["executed_notional_b"]):
            errors.append(f"Non-finite notionals for ticket {d['ticket_id']}")

        gross = abs(d["executed_notional_a"]) + abs(d["executed_notional_b"])

        ticket = next((t for t in tickets if t.ticket_id == d["ticket_id"]), None)
        if ticket:
            if gross > ticket.max_gross_notional * 1.001:
                errors.append(
                    f"Gross notional {gross:.0f} exceeds max {ticket.max_gross_notional:.0f} "
                    f"for ticket {d['ticket_id']}"
                )
            if abs(d["executed_notional_a"]) > ticket.max_leg_notional * 1.001:
                errors.append(
                    f"Leg A notional exceeds max for ticket {d['ticket_id']}"
                )
            if abs(d["executed_notional_b"]) > ticket.max_leg_notional * 1.001:
                errors.append(
                    f"Leg B notional exceeds max for ticket {d['ticket_id']}"
                )

    return errors


def run_test_harness(config: Config) -> bool:
    print("Running test harness...")

    try:
        ensure_directories(config)
        print("  [OK] Directory structure created")
    except Exception as e:
        print(f"  [FAIL] Directory creation: {e}")
        return False

    from src.universe import load_universe
    try:
        universe = load_universe(config)
        print(f"  [OK] Universe loaded: {len(universe.all_symbols)} symbols")
    except Exception as e:
        print(f"  [FAIL] Universe load: {e}")
        return False

    from src.tickets import load_active_tickets
    try:
        test_date = date.today()
        tickets = load_active_tickets(config, test_date)
        print(f"  [OK] Tickets loaded: {len(tickets)} active")
    except Exception as e:
        print(f"  [FAIL] Tickets load: {e}")
        return False

    print("Test harness completed successfully")
    return True
