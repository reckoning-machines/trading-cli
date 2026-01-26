# Claude Instructions: Banks Pair-Trading Controller (Parquet + Polars, FMP Source)

## Objective
Build a Parquet-first, deterministic banks-sector pair-trading controller that supports human-in-the-loop tickets and produces daily recommended target positions with disciplined timing and sizing. Use Financial Modeling Prep (FMP) as the sole market data source for v0. Do not use Postgres in v0.

The system must be runnable entirely from a command line interface. Tickets are authored as YAML files. All outputs must be written to versioned Parquet datasets and a run-specific artifact directory.

## Hard Constraints
- No web UI in v0. Command line only.
- No news ingestion. Do not build any headline workflows.
- Deterministic replay is mandatory: fixed seeds, immutable run directories, full manifests.
- Store all data and logs as Parquet (and a small number of JSON/CSV artifacts as noted).
- No hidden state: every run must be reconstructable from the run directory.
- Add a requirements.txt at repo root.
- Add an architecture.md at repo root describing the system and data layout.
- Do not include any backticks anywhere in the files you produce.

## Deliverables (Repo Structure)
Create the following top-level structure:

- requirements.txt
- architecture.md
- README.md
- banksctl.py (CLI entrypoint)
- src/
  - config.py
  - fmp_client.py
  - storage.py
  - tickets.py
  - universe.py
  - features_pair_state.py
  - risk_governor.py
  - controller_baseline.py
  - controller_rl_stub.py
  - run_engine.py
  - reporting.py
- tickets/
  - active/
  - archived/
- data/
  - fmp_cache/
    - daily_bars/
    - meta/
  - features/
    - pair_state/
- runs/

## Dependencies (requirements.txt)
requirements.txt must include at least:
- polars
- pyarrow
- pandas
- pyyaml
- requests
- python-dateutil
- numpy
- rich
- typer

If you add RL in v0 as a stub, do not add heavy RL libraries. The RL controller can be a placeholder module with an interface identical to the baseline controller.

## Configuration
Implement src/config.py to load configuration from environment variables with safe defaults:
- FMP_API_KEY (required)
- DATA_DIR (default ./data)
- RUNS_DIR (default ./runs)
- TICKETS_DIR (default ./tickets/active)
- FEATURE_VERSION (default v1)
- UNIVERSE_FILE (default ./universe.yaml)

No secrets in files. Only environment variables.

## Data Model and Storage Layout
All file paths must be under DATA_DIR and RUNS_DIR. Use the following canonical layout.

1) FMP cache, daily bars:
- data/fmp_cache/daily_bars/symbol=JPM/part-0000.parquet
- data/fmp_cache/daily_bars/symbol=BAC/part-0000.parquet

Schema for cached bars:
- date (date)
- symbol (string)
- adj_close (float64)
- close (float64)
- volume (int64)

2) FMP pull log:
- data/fmp_cache/meta/fmp_pull_log.parquet

Schema:
- pulled_at_ts (datetime)
- symbol (string)
- start_date (date)
- end_date (date)
- row_count (int64)
- data_checksum (string)
- source (string, always FMP)

3) Pair state features:
- data/features/pair_state/version=v1/asof_date=2026-01-26/part-0000.parquet

Schema:
- asof_date (date)
- pair_id (string, e.g. JPM__BAC)
- leg_a (string)
- leg_b (string)
- window_hedge (int64)
- window_z (int64)
- beta (float64)
- spread (float64)
- zscore (float64)
- beta_stability (float64)
- no_mean_cross_days (int64)
- extreme_z (int8)
- feature_version (string)

4) Runs
Create a run directory per invocation:
- runs/run_id=YYYY-MM-DDTHHMMSSZ_<short_hash>/

Files in each run directory:
- manifest.json
- tickets_snapshot/ (copy of ticket YAMLs used)
- decisions.parquet
- blotter.csv
- metrics.json
- logs.txt

Decisions schema:
- run_id (string)
- asof_date (date)
- ticket_id (string)
- pair_id (string)
- proposed_units (float64)
- final_units (float64)
- clamp_codes (string, comma separated)
- notional_a (float64)
- notional_b (float64)
- est_costs (float64)
- est_daily_vol_dollars (float64)
- feature_version (string)
- controller_version (string)

Blotter CSV must include at least:
- ticket_id, pair_id, leg_a, leg_b, notional_a, notional_b, final_units, clamp_codes

Manifest JSON must include:
- run_id
- asof_ts
- asof_date
- git_sha (if available)
- seed
- feature_version
- controller_version
- universe_symbols
- ticket_files_used (paths)
- fmp_cache_state_summary (max date per symbol)

## Ticketing (Human-in-the-loop)
Tickets are YAML files under tickets/active.

Implement src/tickets.py with:
- PairTicket model with explicit validation.

Ticket YAML required fields:
- ticket_id
- type (must be PAIR)
- leg_a
- leg_b
- orientation (must be LONG_SPREAD or SHORT_SPREAD)
- max_gross_notional (float, dollars)
- max_leg_notional (float, dollars)
- horizon_days (int)
- conviction (int 1 to 5)
- entry_z (float, optional)
- exit_z (float, optional)
- time_stop_days (int)
- flatten_on_breakdown (bool)
- expires_on (date string)

Ticket parsing rules:
- Normalize symbols to uppercase.
- pair_id is leg_a__leg_b
- Reject tickets with expires_on earlier than asof_date.

## Universe
Implement src/universe.py:
- Load a universe.yaml with a list of tickers and optional pair suggestions.
- If no universe.yaml exists, default to a small list: JPM, BAC, WFC, C, GS, MS, SCHW, PNC plus XLF and KRE for context only.

Pairs:
- For v0, only compute features and decisions for pairs referenced by active tickets.

## CLI (banksctl.py)
Use Typer and Rich.

Commands required:

1) ingest-prices
- banksctl ingest-prices --start YYYY-MM-DD --end YYYY-MM-DD
Behavior:
- For each symbol in universe:
  - Pull daily adjusted bars from FMP.
  - Append to Parquet dataset partitioned by symbol.
  - Write a pull log row with checksum.
- Print a summary table: symbol, rows pulled, last date.

2) build-features
- banksctl build-features --asof YYYY-MM-DD --feature-version v1 --window-hedge 60 --window-z 60
Behavior:
- Load active tickets to determine required pairs.
- Load cached prices for leg_a and leg_b.
- Compute rolling beta, spread, zscore as of the asof_date.
- Compute breakdown proxies:
  - beta_stability as rolling standard deviation of beta over window-hedge
  - no_mean_cross_days as days since zscore crossed 0
  - extreme_z as 1 if abs(zscore) >= 3.0 else 0
- Write features parquet partition under the required folder.
- Print a summary table: pair_id, beta, zscore, extreme_z.

3) run-controller
- banksctl run-controller --asof YYYY-MM-DD --feature-version v1 --controller baseline --seed 123
Behavior:
- Create run_id and run directory.
- Snapshot tickets into tickets_snapshot.
- Load pair_state features for asof_date and feature_version.
- For each active ticket:
  - Compute proposed target units using controller_baseline.
  - Convert units to leg notionals using hedge ratio beta and ticket max_gross_notional:
    - unit range is [-1.0, -0.5, 0.0, 0.5, 1.0]
    - gross notional = abs(notional_a) + abs(notional_b) must not exceed max_gross_notional
    - use notional_a = sign(unit) * N
    - use notional_b = -sign(unit) * beta * N for LONG_SPREAD under spread definition
    - adjust sign based on orientation
    - choose N so that gross notional equals abs(unit) times max_gross_notional before clamps
  - Apply RiskGovernor clamps:
    - per-leg cap max_leg_notional
    - gross cap max_gross_notional
    - if flatten_on_breakdown and extreme_z==1 then force final_units=0 and clamp code BREAKDOWN_FLATTEN
    - if ticket expired then skip
  - Estimate costs as a simple linear cost: cost_bps * gross_notional, with cost_bps default 1.0
  - Estimate daily vol dollars as abs(unit) * max_gross_notional * vol_proxy, where vol_proxy is computed from recent spread returns standard deviation
- Write decisions.parquet, blotter.csv, metrics.json, logs.txt, manifest.json.
- Print a clear terminal summary:
  - actions by ticket
  - clamp codes
  - notionals

4) report
- banksctl report --run-id <RUN_ID>
Behavior:
- Load decisions.parquet and print summary.
- Load metrics.json if present.

## Baseline Controller (controller_baseline.py)
Implement a deterministic policy:
- If abs(zscore) < exit_z then target units = 0
- Else if abs(zscore) >= entry_z then target units = sign(zscore) * -1.0 for mean reversion unless orientation specifies otherwise
- Use conviction to scale units:
  - conviction 1 to 2 yields max 0.5
  - conviction 3 to 5 yields max 1.0
- Clamp to allowed discrete set [-1.0, -0.5, 0.0, 0.5, 1.0]

If entry_z and exit_z are missing in ticket, default entry_z=1.5 and exit_z=0.2.

## Risk Governor (risk_governor.py)
Implement a pure function:
- input: ticket, beta, proposed_units, proposed notionals, feature flags
- output: final_units, final notionals, clamp_codes list

Clamp code taxonomy must include:
- GROSS_CLAMP
- LEG_CLAMP_A
- LEG_CLAMP_B
- BREAKDOWN_FLATTEN
- EXPIRED_SKIP

## Architecture Document (architecture.md)
Write an explicit architecture.md that includes:
- system overview in one page
- CLI workflow diagram in text
- data layout and schemas
- determinism and replay rules
- how tickets flow into decisions
- how to add RL later without changing storage contracts

## Determinism and Testing
- Add a seed to all stochastic calls even if baseline uses none.
- Implement a small test harness script under src/run_engine.py that can run a full day pipeline end-to-end.
- Add basic checks:
  - no missing price data at asof_date for any ticket legs
  - beta and zscore are finite
  - gross and leg notionals respect caps after governor
  - run directory contains all required artifacts

## Notes
- Keep implementation explicit and readable.
- No hidden magic.
- Prefer small, well-named functions over clever one-liners.
- Do not implement live execution integration in v0.
- Keep code style professional and suitable for external readers.
