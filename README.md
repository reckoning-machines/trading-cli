# Banks Pair-Trading Controller

A deterministic, Parquet-first pair-trading controller for bank-sector equities with human-in-the-loop ticket management.

## Overview

This system generates daily recommended target positions for pair trades based on:
- Z-score mean reversion signals
- Human-authored trading tickets (YAML files)
- Risk limits and breakdown detection

All data is stored in versioned Parquet datasets. Every run produces a complete artifact directory that enables deterministic replay.

## Installation

    pip install -r requirements.txt

## Configuration

Set required environment variables:

    export FMP_API_KEY=your_api_key_here

Optional configuration:

    export DATA_DIR=./data          # Default: ./data
    export RUNS_DIR=./runs          # Default: ./runs
    export TICKETS_DIR=./tickets/active   # Default: ./tickets/active
    export FEATURE_VERSION=v1       # Default: v1
    export UNIVERSE_FILE=./universe.yaml  # Default: ./universe.yaml

## Quick Start

1. Ingest price data:

    python banksctl.py ingest-prices --start 2025-01-01 --end 2026-01-26

2. Create a ticket (see tickets/active/ for examples):

    Create a YAML file in tickets/active/

3. Build features:

    python banksctl.py build-features --asof 2026-01-26

4. Run the controller:

    python banksctl.py run-controller --asof 2026-01-26 --seed 123

5. Compute realized PnL:

    python banksctl.py realize --asof 2026-01-26 --run-id RUN_ID

6. View the report:

    python banksctl.py report RUN_ID

## CLI Commands

### ingest-prices

Fetch daily price data from Financial Modeling Prep for all universe symbols.

    python banksctl.py ingest-prices --start YYYY-MM-DD --end YYYY-MM-DD

### build-features

Compute pair state features (beta, spread, z-score) for active tickets.

    python banksctl.py build-features --asof YYYY-MM-DD [--feature-version v1] [--window-hedge 60] [--window-z 60]

### run-controller

Generate position recommendations using the specified controller.

    python banksctl.py run-controller --asof YYYY-MM-DD [--feature-version v1] [--controller baseline] [--seed 123]

### report

Display a summary of a completed run.

    python banksctl.py report RUN_ID

### list-runs

Show all available run directories.

    python banksctl.py list-runs

### list-tickets

Show all active tickets.

    python banksctl.py list-tickets

### test-harness

Validate system setup and configuration.

    python banksctl.py test-harness

### realize

Compute realized PnL labels for a completed run.

    python banksctl.py realize --asof YYYY-MM-DD --run-id RUN_ID [--cost-bps 1.0] [--label-version v1]

Outputs:
- runs/run_id=RUN_ID/realized.parquet (run artifact)
- data/labels/realized/version=v1/asof_date=YYYY-MM-DD/ (canonical labels)

### train-policy

Train an RL policy using contextual bandit with ridge regression.

    python banksctl.py train-policy --train-start YYYY-MM-DD --train-end YYYY-MM-DD \
        --eval-start YYYY-MM-DD --eval-end YYYY-MM-DD \
        --feature-version v1 --label-version v1 --model-version v1 \
        [--seed 42] [--lambda-reg 1.0] [--reward-horizon 5]

Outputs:
- data/models/pair_policy/version=v1/model.json (trained model weights)
- data/models/pair_policy/version=v1/training_report.json (training metrics)
- data/models/pair_policy/version=v1/train_examples.parquet (training data)
- data/models/pair_policy/version=v1/eval_examples.parquet (evaluation data)

### list-models

Show all trained RL models.

    python banksctl.py list-models

### console

Start the interactive prompt console for portfolio management.

    python banksctl.py console

The console supports four intents:
- **SUMMARIZE**: Show strategy overview ("summarize todays strategy", "status")
- **GENERATE_IDEAL_PORTFOLIO**: Generate symbol-level portfolio ("generate todays ideal portfolio")
- **RUN_BACKTEST**: Run historical backtest ("backtest last 6 months")
- **DRAFT_TICKET**: Create ticket drafts ("draft ticket for JPM and BAC")

Portfolio constraints (environment variables):
- PORTFOLIO_MAX_GROSS_NOTIONAL: Maximum total gross notional
- PORTFOLIO_MAX_NAME_GROSS_NOTIONAL: Maximum gross per symbol

The portfolio generator enforces exact net zero exposure by adding a synthetic CASH leg.

## Ticket Format

Tickets are YAML files placed in tickets/active/. Required fields:

    ticket_id: JPM_BAC_001
    type: PAIR
    leg_a: JPM
    leg_b: BAC
    orientation: LONG_SPREAD
    max_gross_notional: 1000000
    max_leg_notional: 600000
    horizon_days: 30
    conviction: 3
    time_stop_days: 20
    flatten_on_breakdown: true
    expires_on: 2026-12-31

Optional fields:

    entry_z: 1.5    # Default: 1.5
    exit_z: 0.2     # Default: 0.2

## Orientation

- LONG_SPREAD: Buy leg_a, sell leg_b when spread is low
- SHORT_SPREAD: Sell leg_a, buy leg_b when spread is low

## Output Files

Each run creates a directory: runs/run_id=TIMESTAMP_HASH/

Contents:
- manifest.json: Run metadata for reproducibility
- tickets_snapshot/: Copies of tickets used
- decisions.parquet: Position recommendations
- blotter.csv: Human-readable summary
- metrics.json: Aggregate statistics
- logs.txt: Execution log
- realized.parquet: Realized PnL labels (after running realize command)

## Data Layout

    data/
      fmp_cache/
        daily_bars/
          symbol=JPM/part-0000.parquet
          symbol=BAC/part-0000.parquet
        meta/
          fmp_pull_log.parquet
      features/
        pair_state/
          version=v1/
            asof_date=2026-01-26/part-0000.parquet
      labels/
        realized/
          version=v1/
            asof_date=2026-01-26/part-0000.parquet
      models/
        pair_policy/
          version=v1/
            model.json
            training_report.json
            train_examples.parquet
            eval_examples.parquet
      state/
        last_positions/
          positions.parquet

## Controllers

### baseline

Deterministic mean-reversion policy:
- Enters position when abs(z-score) >= entry_z
- Exits position when abs(z-score) < exit_z
- Position size scaled by conviction (1-2: half, 3-5: full)
- Discrete positions: -1.0, -0.5, 0.0, 0.5, 1.0

### rl (stub)

Placeholder for reinforcement learning controller. Currently returns no position.

## Risk Governor

Enforces position limits:
- Per-leg notional cap (max_leg_notional)
- Gross notional cap (max_gross_notional)
- Breakdown flattening (if flatten_on_breakdown and extreme_z)
- Ticket expiration

## Architecture

See architecture.md for detailed system documentation.
