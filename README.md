# Trading CLI
### A Deterministic Portfolio Construction and Risk Evaluation Engine for Bank-Sector Pair Strategies

---

## Overview

Trading CLI is a deterministic, Parquet-first **risk evaluation and portfolio construction system** designed to support **human-directed trading strategies** in bank-sector equity pairs.

In this system, **humans define the trades**.

Portfolio managers and researchers explicitly specify:
- Which pairs to trade
- The direction of the trade (spread orientation)
- The maximum capital at risk
- The time horizon and invalidation rules

These intentions are expressed as **strategy tickets**.

Trading CLI does **not** originate trades, discover strategies, or exercise discretion.  
Instead, it evaluates **how a human-defined strategy should be expressed today**, given current market state and hard risk constraints.

Specifically, the system:
- Evaluates whether a strategy is currently in-range for entry or exit
- Determines appropriate **position sizing** within predefined budgets
- Enforces ticket-level and portfolio-level risk limits mechanically
- Produces a **daily ideal target portfolio**, not orders

---

## Division of Responsibility

**Humans are responsible for:**
- Strategy selection (what to trade)
- Structural assumptions (why the trade exists)
- Capital budgets and risk tolerance
- Strategy lifecycle (activation and expiration)

**Trading CLI is responsible for:**
- Evaluating entry and exit conditions
- Scaling exposure based on current signal strength
- Enforcing per-leg, per-ticket, per-name, and portfolio-wide constraints
- Ensuring exact net neutrality and conservative capital usage
- Producing reproducible, auditable portfolio artifacts

This separation ensures that **judgment remains human**, while **discipline is mechanical**.

---

## Outputs

Each run produces:
- A symbol-level **ideal portfolio** with exact net zero exposure
- Explicit attribution from symbols back to strategy tickets
- Detailed clamp and risk-enforcement logs
- Immutable artifacts suitable for review, audit, and backtesting

Trading CLI exists to answer a single question:

**“Given the strategies I believe in, what is the safest and most disciplined way to express them today?”**

---

## What This System Is (and Is Not)

This system is:
- A portfolio construction engine
- A mechanical risk governor
- A strategy expression and validation layer
- A reproducible research and backtesting system

This system is not:
- An execution engine
- An intraday trading system
- A signal discovery engine
- An autonomous trading agent

---

## Conceptual Workflow

Market data, strategy tickets, and universe definitions flow into a deterministic pipeline:

```
                           INPUTS
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │    FMP API      │  │    Tickets      │  │    Universe     │
    │   (prices)      │  │    (YAML)       │  │    (YAML)       │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             ▼                    │                    │
    ┌─────────────────┐           │                    │
    │  ingest-prices  │           │                    │
    └────────┬────────┘           │                    │
             │                    │                    │
             ▼                    │                    │
    ┌─────────────────┐           │                    │
    │  data/fmp_cache │           │                    │
    │   daily_bars/   │           │                    │
    └────────┬────────┘           │                    │
             │                    │                    │
             └────────────┬───────┴────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ build-features  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ data/features/  │
                 │  pair_state/    │
                 └────────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐  ┌─────────────┐  ┌───────────┐
    │  console  │  │run-controller│  │  realize  │
    └─────┬─────┘  └──────┬──────┘  └─────┬─────┘
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐  ┌─────────────┐  ┌───────────┐
    │  Ideal    │  │ decisions/  │  │ realized  │
    │ Portfolio │  │  blotter    │  │   PnL     │
    │ + CASH    │  └─────────────┘  └─────┬─────┘
    │ Backtest  │                         │
    │  Drafts   │                         ▼
    └───────────┘                  ┌─────────────┐
                                   │train-policy │
                                   └──────┬──────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │ model.json  │
                                   │ (RL policy) │
                                   └─────────────┘
```

Pipeline stages:
- Price data is ingested and cached locally
- Pair-state features are computed as of a given date
- Active tickets express human strategy intent and risk budgets
- A controller proposes ticket-level target exposures
- Risk governors enforce ticket-level and portfolio-level constraints
- An ideal portfolio is constructed with exact net zero exposure
- Backtests evaluate historical performance under identical rules

An interactive console sits above the pipeline to summarize strategy state, generate today's ideal portfolio, run backtests, and draft tickets.

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Configuration

Required environment variable:

export FMP_API_KEY=your_api_key_here

Optional configuration:

export DATA_DIR=./data  
export RUNS_DIR=./runs  
export TICKETS_DIR=./tickets/active  
export FEATURE_VERSION=v1  
export UNIVERSE_FILE=./universe.yaml  

Portfolio-level constraints:

export PORTFOLIO_MAX_GROSS_NOTIONAL=5000000  
export PORTFOLIO_MAX_NAME_GROSS_NOTIONAL=1000000  

---

## Quick Start

1. Ingest price data:

python banksctl.py ingest-prices --start 2025-01-01 --end 2026-01-26

2. Create a strategy ticket:

Create a YAML file in tickets/active

3. Build pair features:

python banksctl.py build-features --asof 2026-01-26

4. Generate today’s ideal portfolio:

python banksctl.py console  
generate todays ideal portfolio

5. Run a historical backtest:

backtest last 6 months

---

## Interactive Console

Start the console:

python banksctl.py console

Supported intents:
- SUMMARIZE: show active strategies and portfolio status
- GENERATE_IDEAL_PORTFOLIO: construct today’s ideal portfolio
- RUN_BACKTEST: evaluate historical performance
- DRAFT_TICKET: create or lint strategy tickets (drafts only)

All console actions follow a plan-then-confirm workflow and never modify active strategies automatically.

---

## Portfolio Semantics

The ideal portfolio obeys the following invariants:

- Per-ticket limits enforced mechanically
- Per-name gross exposure enforced as true gross across tickets
- Portfolio gross enforced across equity symbols only
- Exact net zero exposure enforced via a synthetic CASH leg
- CASH does not count toward equity gross constraints

---

## Backtest Semantics

Backtests:
- Use the same portfolio governor as live portfolio construction
- Use a single common next trading date across all equity symbols
- Skip entire days if any required price data is missing
- Maintain portfolio state across skipped days for accurate turnover
- Report skipped days explicitly in metrics

No partial PnL days are allowed.

---

## Ticket Format

Tickets are YAML files placed in tickets/active.

Required fields:
- ticket_id
- type (PAIR)
- leg_a
- leg_b
- orientation
- max_gross_notional
- max_leg_notional
- horizon_days
- conviction
- time_stop_days
- flatten_on_breakdown
- expires_on

Optional fields:
- entry_z
- exit_z

Tickets express intent and risk budgets only. They do not execute trades.

---

## Controllers

Baseline:
- Deterministic mean-reversion policy
- Discrete exposure levels
- Scaled by conviction

Learned:
- Optional policy model for allocation and sizing
- Operates under identical risk constraints
- Evaluated offline before use

---

## Output Artifacts

Each run creates a directory under runs/ containing:
- manifest.json
- tickets_snapshot
- ideal_portfolio.parquet
- portfolio_blotter.csv
- metrics.json
- logs.txt

All artifacts are immutable and replayable.

---

## Architecture

See architecture.md for detailed system design and invariants.
