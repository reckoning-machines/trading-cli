# Policy Bandit - Contextual bandit with ridge regression for action selection

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import polars as pl

from src.config import Config
from src.training_dataset import get_state_feature_columns, ALLOWED_ACTIONS


class RidgeBanditPolicy:
    # Contextual bandit with separate ridge regression model per action
    # Predicts expected reward for each action given state features

    def __init__(self, lambda_reg: float = 1.0, seed: int = 42):
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Model weights: one set per action
        # weights[action] = weight vector of shape (n_features,)
        self.weights: Dict[float, np.ndarray] = {}

        # Feature statistics for normalization
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        self.feature_columns = get_state_feature_columns()
        self.n_features = len(self.feature_columns)
        self.is_fitted = False
        self.training_stats: Dict[str, Any] = {}

    def fit(self, train_df: pl.DataFrame) -> Dict[str, Any]:
        # Train separate ridge regression for each action
        # X: state features, y: reward_5d

        if train_df.is_empty():
            return {"error": "Empty training data"}

        # Extract features and targets
        X_all = train_df.select(self.feature_columns).to_numpy()
        actions_all = train_df["action_units"].to_numpy()
        rewards_all = train_df["reward_5d"].to_numpy()

        # Compute feature normalization
        self.feature_means = np.nanmean(X_all, axis=0)
        self.feature_stds = np.nanstd(X_all, axis=0)
        # Avoid division by zero
        self.feature_stds[self.feature_stds < 1e-8] = 1.0

        # Normalize features
        X_norm = (X_all - self.feature_means) / self.feature_stds

        # Handle any remaining NaNs
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        # Train a model for each action
        stats_per_action = {}

        for action in ALLOWED_ACTIONS:
            # Filter examples where this action was taken
            mask = np.abs(actions_all - action) < 0.01
            X_action = X_norm[mask]
            y_action = rewards_all[mask]

            n_samples = len(y_action)

            if n_samples < 2:
                # Not enough data for this action - initialize with zeros
                self.weights[action] = np.zeros(self.n_features)
                stats_per_action[action] = {
                    "n_samples": n_samples,
                    "mean_reward": 0.0,
                    "status": "insufficient_data",
                }
                continue

            # Ridge regression: w = (X^T X + lambda I)^-1 X^T y
            XtX = X_action.T @ X_action
            Xty = X_action.T @ y_action
            I = np.eye(self.n_features)

            try:
                w = np.linalg.solve(XtX + self.lambda_reg * I, Xty)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                w = np.linalg.lstsq(XtX + self.lambda_reg * I, Xty, rcond=None)[0]

            self.weights[action] = w

            # Compute training statistics
            y_pred = X_action @ w
            mse = np.mean((y_action - y_pred) ** 2)
            mean_reward = np.mean(y_action)

            stats_per_action[action] = {
                "n_samples": n_samples,
                "mean_reward": float(mean_reward),
                "mse": float(mse),
                "status": "ok",
            }

        self.is_fitted = True
        self.training_stats = {
            "total_samples": len(actions_all),
            "lambda_reg": self.lambda_reg,
            "n_features": self.n_features,
            "action_stats": stats_per_action,
        }

        return self.training_stats

    def predict_action_values(self, state_features: np.ndarray) -> Dict[float, float]:
        # Predict expected reward for each action given state features
        # state_features: shape (n_features,)

        if not self.is_fitted:
            return {a: 0.0 for a in ALLOWED_ACTIONS}

        # Normalize features
        x_norm = (state_features - self.feature_means) / self.feature_stds
        x_norm = np.nan_to_num(x_norm, nan=0.0)

        values = {}
        for action in ALLOWED_ACTIONS:
            if action in self.weights:
                values[action] = float(np.dot(self.weights[action], x_norm))
            else:
                values[action] = 0.0

        return values

    def select_action(self, state_features: np.ndarray, greedy: bool = True) -> float:
        # Select action based on predicted values
        # greedy=True: argmax, greedy=False: sample proportional to softmax

        values = self.predict_action_values(state_features)

        if greedy:
            # Deterministic argmax with tie-breaking by action order
            best_action = max(ALLOWED_ACTIONS, key=lambda a: (values[a], -abs(a)))
            return best_action
        else:
            # Softmax sampling
            vals = np.array([values[a] for a in ALLOWED_ACTIONS])
            # Temperature scaling for exploration
            temperature = 1.0
            vals = vals / temperature
            # Numerical stability
            vals = vals - np.max(vals)
            probs = np.exp(vals)
            probs = probs / np.sum(probs)

            action_idx = self.rng.choice(len(ALLOWED_ACTIONS), p=probs)
            return ALLOWED_ACTIONS[action_idx]

    def save(self, model_dir: Path) -> None:
        # Save model weights and metadata to directory
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save weights as numpy arrays
        weights_dict = {}
        for action, w in self.weights.items():
            weights_dict[str(action)] = w.tolist()

        model_data = {
            "weights": weights_dict,
            "feature_means": self.feature_means.tolist() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
            "lambda_reg": self.lambda_reg,
            "seed": self.seed,
            "feature_columns": self.feature_columns,
            "n_features": self.n_features,
            "is_fitted": self.is_fitted,
            "training_stats": self.training_stats,
            "saved_at": datetime.utcnow().isoformat(),
        }

        model_path = model_dir / "model.json"
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path) -> "RidgeBanditPolicy":
        # Load model from directory
        model_path = model_dir / "model.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, "r") as f:
            model_data = json.load(f)

        policy = cls(
            lambda_reg=model_data.get("lambda_reg", 1.0),
            seed=model_data.get("seed", 42),
        )

        # Restore weights
        weights_dict = model_data.get("weights", {})
        for action_str, w_list in weights_dict.items():
            action = float(action_str)
            policy.weights[action] = np.array(w_list)

        # Restore normalization stats
        if model_data.get("feature_means") is not None:
            policy.feature_means = np.array(model_data["feature_means"])
        if model_data.get("feature_stds") is not None:
            policy.feature_stds = np.array(model_data["feature_stds"])

        policy.is_fitted = model_data.get("is_fitted", False)
        policy.training_stats = model_data.get("training_stats", {})

        return policy


def evaluate_policy(
    policy: RidgeBanditPolicy,
    eval_df: pl.DataFrame,
    baseline_action_col: str = "action_units",
) -> Dict[str, Any]:
    # Evaluate policy on held-out data
    # Compare learned policy actions vs baseline actions

    if eval_df.is_empty():
        return {"error": "Empty evaluation data"}

    feature_columns = get_state_feature_columns()
    X = eval_df.select(feature_columns).to_numpy()
    baseline_actions = eval_df[baseline_action_col].to_numpy()
    rewards = eval_df["reward_5d"].to_numpy()

    n_samples = len(rewards)

    # Compute baseline performance (what actually happened)
    baseline_total_reward = np.sum(rewards)
    baseline_mean_reward = np.mean(rewards)

    # Compute learned policy actions
    learned_actions = []
    learned_values = []

    for i in range(n_samples):
        state = X[i]
        action = policy.select_action(state, greedy=True)
        learned_actions.append(action)
        values = policy.predict_action_values(state)
        learned_values.append(values.get(action, 0.0))

    learned_actions = np.array(learned_actions)

    # Count action distribution
    baseline_action_counts = {}
    learned_action_counts = {}

    for action in ALLOWED_ACTIONS:
        baseline_action_counts[action] = int(np.sum(np.abs(baseline_actions - action) < 0.01))
        learned_action_counts[action] = int(np.sum(np.abs(learned_actions - action) < 0.01))

    # Agreement rate
    agreement = np.sum(np.abs(baseline_actions - learned_actions) < 0.01)
    agreement_rate = float(agreement / n_samples) if n_samples > 0 else 0.0

    # Predicted reward (counterfactual)
    predicted_mean_reward = np.mean(learned_values)

    return {
        "n_samples": n_samples,
        "baseline_total_reward": float(baseline_total_reward),
        "baseline_mean_reward": float(baseline_mean_reward),
        "predicted_mean_reward": float(predicted_mean_reward),
        "agreement_rate": agreement_rate,
        "baseline_action_counts": baseline_action_counts,
        "learned_action_counts": learned_action_counts,
    }


def walk_forward_evaluation(
    config: Config,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    lambda_reg: float = 1.0,
    seed: int = 42,
) -> Tuple[RidgeBanditPolicy, Dict[str, Any]]:
    # Train on train_df, evaluate on eval_df

    policy = RidgeBanditPolicy(lambda_reg=lambda_reg, seed=seed)

    # Train
    train_stats = policy.fit(train_df)

    # Evaluate
    eval_stats = evaluate_policy(policy, eval_df)

    return policy, {
        "train_stats": train_stats,
        "eval_stats": eval_stats,
    }
