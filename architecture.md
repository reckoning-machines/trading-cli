# Banks Pair-Trading Controller Architecture

## System Overview

The Banks Pair-Trading Controller is a deterministic, Parquet-first system for generating daily recommended target positions for bank-sector equity pair trades. The system operates entirely via command line interface and supports human-in-the-loop decision making through YAML ticket files.

Key design principles:
- Deterministic replay: Every run can be reconstructed from the run directory
- No hidden state: All data and decisions are persisted to versioned Parquet files
- Human-in-the-loop: Trading ideas are authored as YAML tickets
- Separation of concerns: Feature computation, control policy, and risk governance are distinct modules

## CLI Workflow Diagram

    +------------------+
    | universe.yaml    |
    | (symbol list)    |
    +--------+---------+
             |
             v
    +------------------+     +------------------+
    | ingest-prices    |---->| data/fmp_cache/  |
    | (FMP API)        |     | daily_bars/      |
    +------------------+     +------------------+
                                     |
                                     v
    +------------------+     +------------------+
    | tickets/active/  |---->| build-features   |
    | (YAML files)     |     | (pair state)     |
    +------------------+     +--------+---------+
                                     |
                                     v
                             +------------------+
                             | data/features/   |
                             | pair_state/      |
                             +--------+---------+
                                     |
                                     v
                             +------------------+
                             | run-controller   |
                             | (decisions)      |
                             +--------+---------+
                                     |
                                     v
                             +------------------+
                             | runs/run_id=.../ |
                             | (artifacts)      |
                             +--------+---------+
                                     |
                                     v
                             +------------------+
                             | realize          |
                             | (PnL labels)     |
                             +--------+---------+
                                     |
                                     v
                             +------------------+
                             | data/labels/     |
                             | realized/        |
                             +------------------+

## Data Layout and Schemas

### FMP Cache - Daily Bars

Path: data/fmp_cache/daily_bars/symbol=SYMBOL/part-0000.parquet

Schema:
- date (date): Trading date
- symbol (string): Ticker symbol
- adj_close (float64): Adjusted closing price
- close (float64): Unadjusted closing price
- volume (int64): Trading volume

### FMP Pull Log

Path: data/fmp_cache/meta/fmp_pull_log.parquet

Schema:
- pulled_at_ts (datetime): UTC timestamp of pull
- symbol (string): Ticker symbol
- start_date (date): Requested start date
- end_date (date): Requested end date
- row_count (int64): Number of rows returned
- data_checksum (string): SHA256 checksum (first 16 chars)
- source (string): Always "FMP"

### Pair State Features

Path: data/features/pair_state/version=VERSION/asof_date=DATE/part-0000.parquet

Schema:
- asof_date (date): Feature computation date
- pair_id (string): Pair identifier (e.g., JPM__BAC)
- leg_a (string): First leg symbol
- leg_b (string): Second leg symbol
- window_hedge (int64): Rolling window for beta
- window_z (int64): Rolling window for z-score
- beta (float64): Hedge ratio
- spread (float64): Log spread value
- zscore (float64): Standardized spread
- beta_stability (float64): Rolling std of beta
- no_mean_cross_days (int64): Days since z-score crossed zero
- extreme_z (int8): 1 if abs(zscore) >= 3.0, else 0
- feature_version (string): Version identifier

### Run Artifacts

Path: runs/run_id=TIMESTAMP_HASH/

Files:
- manifest.json: Run metadata and reproducibility info
- tickets_snapshot/: Copy of YAML tickets used
- decisions.parquet: Position recommendations
- blotter.csv: Human-readable position summary
- metrics.json: Aggregate run metrics
- logs.txt: Execution log

Decisions Schema:
- run_id (string): Run identifier
- asof_date (date): Decision date
- ticket_id (string): Source ticket
- pair_id (string): Pair identifier
- proposed_units (float64): Controller output before clamps
- final_units (float64): Post-governor position
- clamp_codes (string): Comma-separated clamp reasons
- notional_a (float64): Dollar position in leg A
- notional_b (float64): Dollar position in leg B
- est_costs (float64): Estimated transaction costs
- est_daily_vol_dollars (float64): Estimated daily P&L volatility
- feature_version (string): Feature version used
- controller_version (string): Controller version used

### Realized PnL Labels

Run artifact path: runs/run_id=TIMESTAMP_HASH/realized.parquet
Canonical path: data/labels/realized/version=VERSION/asof_date=DATE/part-0000.parquet

Schema:
- run_id (string): Run identifier
- asof_date (date): Decision date
- next_date (date): Next trading date for return calculation
- ticket_id (string): Source ticket
- pair_id (string): Pair identifier
- leg_a (string): First leg symbol
- leg_b (string): Second leg symbol
- executed_notional_a (float64): Executed position in leg A
- executed_notional_b (float64): Executed position in leg B
- prev_notional_a (float64): Previous position in leg A
- prev_notional_b (float64): Previous position in leg B
- trade_notional (float64): Absolute turnover
- gross_exposure (float64): Total absolute notional
- ret_a (float64): Return on leg A
- ret_b (float64): Return on leg B
- pnl_gross (float64): Gross PnL before costs
- costs (float64): Transaction costs
- pnl_net (float64): Net PnL after costs
- cost_bps (float64): Cost rate in basis points
- label_version (string): Label version identifier
- status (string): OK or NO_PRICE_SKIP

### Position State Store

Path: data/state/last_positions/positions.parquet

Schema:
- ticket_id (string): Ticket identifier
- last_notional_a (float64): Last executed notional in leg A
- last_notional_b (float64): Last executed notional in leg B
- last_asof_date (date): Date of last execution

## Determinism and Replay Rules

1. Fixed Seeds: All stochastic operations use explicit seeds passed as CLI arguments

2. Immutable Run Directories: Once created, run directories are never modified

3. Full Manifests: Every run records:
   - Exact timestamps
   - Git SHA (if available)
   - All input file paths
   - FMP cache state summary
   - Random seed

4. Ticket Snapshots: Tickets are copied to the run directory at execution time

5. Replay Procedure:
   - Load manifest.json from target run
   - Ensure FMP cache contains data through asof_date
   - Restore tickets from tickets_snapshot/
   - Run controller with same seed and feature_version
   - Output should match decisions.parquet exactly

## Ticket Flow Into Decisions

    +------------------+
    | Ticket YAML      |
    | - ticket_id      |
    | - leg_a, leg_b   |
    | - orientation    |
    | - limits         |
    +--------+---------+
             |
             | load_active_tickets()
             v
    +------------------+
    | PairTicket       |
    | (validated)      |
    +--------+---------+
             |
             | pair_id = leg_a__leg_b
             v
    +------------------+
    | Pair Features    |
    | - beta, zscore   |
    | - extreme_z      |
    +--------+---------+
             |
             | controller.compute_target_units()
             v
    +------------------+
    | Proposed Units   |
    | [-1, -0.5, 0,    |
    |  0.5, 1]         |
    +--------+---------+
             |
             | controller.compute_notionals()
             v
    +------------------+
    | Proposed         |
    | Notionals        |
    +--------+---------+
             |
             | risk_governor.apply_risk_governor()
             v
    +------------------+
    | Final Decision   |
    | - final_units    |
    | - final notionals|
    | - clamp_codes    |
    +------------------+

## Clamp Code Taxonomy

- GROSS_CLAMP: Total position reduced to meet max_gross_notional
- LEG_CLAMP_A: Leg A position reduced to meet max_leg_notional
- LEG_CLAMP_B: Leg B position reduced to meet max_leg_notional
- BREAKDOWN_FLATTEN: Position zeroed due to extreme_z and flatten_on_breakdown
- EXPIRED_SKIP: Ticket expired, no action taken

## Adding RL Without Changing Storage Contracts

The RL controller stub (controller_rl_stub.py) provides the same interface as the baseline controller:

    def compute_target_units(ticket, zscore, orientation, seed) -> float
    def compute_notionals(units, beta, max_gross_notional, orientation) -> Tuple[float, float]
    def get_controller_version() -> str

To implement a full RL controller:

1. Replace the stub logic in compute_target_units with:
   - State construction from features
   - Policy model loading (checkpoint path from config)
   - Inference to get action distribution
   - Action sampling or argmax

2. The output interface remains unchanged:
   - Returns a float in [-1.0, 1.0] range
   - Will be discretized by clamp_to_allowed()
   - Passes through same risk_governor

3. Storage contracts remain identical:
   - Same decisions.parquet schema
   - Same manifest.json structure
   - Only controller_version field changes

4. No changes required to:
   - Feature computation
   - Risk governor
   - Run engine
   - CLI commands
   - Reporting

5. Training pipeline (future work):
   - Read historical features from data/features/pair_state/
   - Define reward function from realized P&L
   - Train offline on historical trajectories
   - Save policy checkpoints to models/ directory
   - Update controller_rl_stub.py to load and serve

## Module Responsibilities

- config.py: Environment variable loading, path configuration
- fmp_client.py: FMP API interaction, data fetching
- storage.py: Parquet I/O, directory management
- tickets.py: YAML parsing, ticket validation
- universe.py: Symbol list management
- features_pair_state.py: Feature computation (beta, spread, zscore)
- risk_governor.py: Position limit enforcement
- controller_baseline.py: Deterministic mean-reversion policy
- controller_rl_stub.py: RL policy placeholder
- run_engine.py: Orchestration, artifact generation
- realized_pnl.py: Realized PnL and cost label computation
- reporting.py: Terminal output formatting
- banksctl.py: CLI entrypoint
