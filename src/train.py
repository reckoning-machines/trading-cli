# Training module - orchestrates policy training and evaluation

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import polars as pl

from src.config import Config
from src.training_dataset import (
    build_training_examples,
    build_training_examples_from_decisions,
    get_state_feature_columns,
    ALLOWED_ACTIONS,
)
from src.policy_bandit import (
    RidgeBanditPolicy,
    evaluate_policy,
    walk_forward_evaluation,
)


def train_policy(
    config: Config,
    train_start: date,
    train_end: date,
    eval_start: date,
    eval_end: date,
    feature_version: str,
    label_version: str,
    model_version: str,
    seed: int = 42,
    lambda_reg: float = 1.0,
    reward_horizon: int = 5,
) -> Dict[str, Any]:
    # Main training function
    # 1. Build training dataset from [train_start, train_end]
    # 2. Build evaluation dataset from [eval_start, eval_end]
    # 3. Train ridge regression policy
    # 4. Evaluate on held-out data
    # 5. Save model and training report

    # Build training examples
    train_df = build_training_examples(
        config=config,
        start_date=train_start,
        end_date=train_end,
        feature_version=feature_version,
        label_version=label_version,
        reward_horizon=reward_horizon,
    )

    # Build evaluation examples
    eval_df = build_training_examples(
        config=config,
        start_date=eval_start,
        end_date=eval_end,
        feature_version=feature_version,
        label_version=label_version,
        reward_horizon=reward_horizon,
    )

    if train_df.is_empty():
        return {
            "error": "No training examples found",
            "train_start": str(train_start),
            "train_end": str(train_end),
        }

    # Train and evaluate
    policy, results = walk_forward_evaluation(
        config=config,
        train_df=train_df,
        eval_df=eval_df,
        lambda_reg=lambda_reg,
        seed=seed,
    )

    # Create model directory
    model_dir = config.pair_policy_dir / f"version={model_version}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    policy.save(model_dir)

    # Save training examples for reproducibility
    train_path = model_dir / "train_examples.parquet"
    train_df.write_parquet(train_path)

    if not eval_df.is_empty():
        eval_path = model_dir / "eval_examples.parquet"
        eval_df.write_parquet(eval_path)

    # Build training report
    report = {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "feature_version": feature_version,
        "label_version": label_version,
        "seed": seed,
        "lambda_reg": lambda_reg,
        "reward_horizon": reward_horizon,
        "n_train_examples": len(train_df),
        "n_eval_examples": len(eval_df),
        "train_stats": results["train_stats"],
        "eval_stats": results["eval_stats"],
        "feature_columns": get_state_feature_columns(),
        "allowed_actions": ALLOWED_ACTIONS,
    }

    # Save report
    report_path = model_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str, sort_keys=True)

    return report


def load_policy(config: Config, model_version: str) -> Optional[RidgeBanditPolicy]:
    # Load a trained policy from disk
    model_dir = config.pair_policy_dir / f"version={model_version}"

    if not model_dir.exists():
        return None

    try:
        return RidgeBanditPolicy.load(model_dir)
    except Exception:
        return None


def list_available_models(config: Config) -> Dict[str, Dict[str, Any]]:
    # List all available trained models
    models = {}

    if not config.pair_policy_dir.exists():
        return models

    for version_dir in sorted(config.pair_policy_dir.iterdir()):
        if version_dir.is_dir() and version_dir.name.startswith("version="):
            version = version_dir.name.replace("version=", "")
            report_path = version_dir / "training_report.json"

            if report_path.exists():
                with open(report_path, "r") as f:
                    report = json.load(f)
                models[version] = {
                    "trained_at": report.get("trained_at"),
                    "n_train_examples": report.get("n_train_examples"),
                    "train_start": report.get("train_start"),
                    "train_end": report.get("train_end"),
                    "eval_stats": report.get("eval_stats", {}),
                }
            else:
                models[version] = {"status": "no_report"}

    return models


def compute_baseline_performance(
    config: Config,
    start_date: date,
    end_date: date,
    feature_version: str,
    label_version: str,
    reward_horizon: int = 5,
) -> Dict[str, Any]:
    # Compute baseline (current controller) performance metrics
    # This uses the actual actions and rewards from the data

    df = build_training_examples(
        config=config,
        start_date=start_date,
        end_date=end_date,
        feature_version=feature_version,
        label_version=label_version,
        reward_horizon=reward_horizon,
    )

    if df.is_empty():
        return {"error": "No data found"}

    rewards = df["reward_5d"].to_numpy()
    actions = df["action_units"].to_numpy()

    # Action distribution
    action_counts = {}
    for action in ALLOWED_ACTIONS:
        import numpy as np
        action_counts[action] = int(np.sum(np.abs(actions - action) < 0.01))

    # Reward statistics
    import numpy as np
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    sharpe = mean_reward / std_reward if std_reward > 0 else 0.0

    return {
        "n_samples": len(df),
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "sharpe": sharpe,
        "action_counts": action_counts,
    }


def compare_policies(
    config: Config,
    model_version: str,
    eval_start: date,
    eval_end: date,
    feature_version: str,
    label_version: str,
    reward_horizon: int = 5,
) -> Dict[str, Any]:
    # Compare learned policy vs baseline on evaluation period

    # Load learned policy
    policy = load_policy(config, model_version)
    if policy is None:
        return {"error": f"Model {model_version} not found"}

    # Build evaluation data
    eval_df = build_training_examples(
        config=config,
        start_date=eval_start,
        end_date=eval_end,
        feature_version=feature_version,
        label_version=label_version,
        reward_horizon=reward_horizon,
    )

    if eval_df.is_empty():
        return {"error": "No evaluation data found"}

    # Evaluate learned policy
    eval_stats = evaluate_policy(policy, eval_df)

    # Compute baseline stats
    baseline_stats = compute_baseline_performance(
        config=config,
        start_date=eval_start,
        end_date=eval_end,
        feature_version=feature_version,
        label_version=label_version,
        reward_horizon=reward_horizon,
    )

    return {
        "model_version": model_version,
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "learned_policy": eval_stats,
        "baseline": baseline_stats,
    }
