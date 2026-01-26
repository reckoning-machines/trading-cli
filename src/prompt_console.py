# Prompt Console - interactive REPL for portfolio generation

import re
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import Config, CASH_SYMBOL, NET_MODE_EXACT_ZERO


class Intent(Enum):
    SUMMARIZE = "SUMMARIZE"
    GENERATE_IDEAL_PORTFOLIO = "GENERATE_IDEAL_PORTFOLIO"
    RUN_BACKTEST = "RUN_BACKTEST"
    DRAFT_TICKET = "DRAFT_TICKET"
    UNKNOWN = "UNKNOWN"
    QUIT = "QUIT"
    HELP = "HELP"


@dataclass
class PromptSession:
    config: Config
    pending_intent: Optional[Intent] = None
    pending_params: Dict[str, Any] = field(default_factory=dict)
    last_plan: Optional[Dict[str, Any]] = None
    last_run_id: Optional[str] = None
    awaiting_confirmation: bool = False
    awaiting_param: Optional[str] = None
    draft_state: Dict[str, Any] = field(default_factory=dict)


# Intent keyword mappings (deterministic routing)
INTENT_KEYWORDS = {
    Intent.SUMMARIZE: [
        "summarize", "summary", "whats my strategy", "what is my strategy",
        "show portfolio", "show strategy", "status", "overview",
        "todays strategy", "today's strategy", "current",
    ],
    Intent.GENERATE_IDEAL_PORTFOLIO: [
        "generate", "build portfolio", "create portfolio", "ideal portfolio",
        "run portfolio", "todays ideal", "today's ideal", "make portfolio",
        "generate portfolio", "build ideal",
    ],
    Intent.RUN_BACKTEST: [
        "backtest", "back test", "evaluate", "historical", "test strategy",
        "run backtest", "simulate",
    ],
    Intent.DRAFT_TICKET: [
        "draft ticket", "create ticket", "new ticket", "write ticket",
        "ticket for", "strategy for", "help me write", "improve ticket",
        "lint ticket", "validate ticket",
    ],
    Intent.QUIT: [
        "quit", "exit", "bye", "q",
    ],
    Intent.HELP: [
        "help", "commands", "what can you do", "?",
    ],
}


def parse_intent(user_text: str) -> Tuple[Intent, Dict[str, Any]]:
    # Parse user input to determine intent and extract parameters
    text_lower = user_text.lower().strip()
    extracted_params = {}

    # Check for quit/help first
    for intent, keywords in [(Intent.QUIT, INTENT_KEYWORDS[Intent.QUIT]),
                              (Intent.HELP, INTENT_KEYWORDS[Intent.HELP])]:
        for keyword in keywords:
            if text_lower == keyword or text_lower.startswith(keyword + " "):
                return intent, extracted_params

    # Check for each intent by keywords
    matched_intent = Intent.UNKNOWN
    best_match_len = 0

    for intent, keywords in INTENT_KEYWORDS.items():
        if intent in (Intent.QUIT, Intent.HELP):
            continue
        for keyword in keywords:
            if keyword in text_lower and len(keyword) > best_match_len:
                matched_intent = intent
                best_match_len = len(keyword)

    # Extract inline parameters
    extracted_params = _extract_inline_params(user_text)

    return matched_intent, extracted_params


def _extract_inline_params(text: str) -> Dict[str, Any]:
    # Extract parameters from user text
    params = {}

    # Date patterns
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    dates = re.findall(date_pattern, text)

    # Check for specific date qualifiers
    text_lower = text.lower()

    if "today" in text_lower:
        params["asof_date"] = date.today()

    if dates:
        if "start" in text_lower and len(dates) >= 1:
            params["start_date"] = date.fromisoformat(dates[0])
        if "end" in text_lower and len(dates) >= 2:
            params["end_date"] = date.fromisoformat(dates[1])
        elif len(dates) == 2:
            params["start_date"] = date.fromisoformat(dates[0])
            params["end_date"] = date.fromisoformat(dates[1])
        elif len(dates) == 1 and "start_date" not in params:
            params["asof_date"] = date.fromisoformat(dates[0])

    # Time period patterns
    month_pattern = r"last\s+(\d+)\s+months?"
    month_match = re.search(month_pattern, text_lower)
    if month_match:
        months = int(month_match.group(1))
        params["lookback_months"] = months

    # Controller patterns
    if "baseline" in text_lower:
        params["controller"] = "baseline"
    elif "rl" in text_lower or "learned" in text_lower:
        params["controller"] = "rl"

    # Symbol patterns (for ticket drafting)
    symbol_pattern = r"\b([A-Z]{1,5})\s+(?:and|vs|versus|/)\s+([A-Z]{1,5})\b"
    symbol_match = re.search(symbol_pattern, text)
    if symbol_match:
        params["leg_a"] = symbol_match.group(1)
        params["leg_b"] = symbol_match.group(2)

    # Yes/No patterns
    if "yes" in text_lower:
        params["confirm"] = True
    elif "no" in text_lower:
        params["confirm"] = False

    return params


def build_plan(
    intent: Intent,
    params: Dict[str, Any],
    config: Config,
) -> Tuple[Dict[str, Any], List[str]]:
    # Build a plan for the given intent
    # Returns (plan_dict, missing_params_list)
    plan = {
        "intent": intent.value,
        "timestamp": datetime.utcnow().isoformat(),
    }
    missing = []

    if intent == Intent.SUMMARIZE:
        plan["action"] = "summarize_strategy"
        # No required params for summarize

    elif intent == Intent.GENERATE_IDEAL_PORTFOLIO:
        plan["action"] = "generate_ideal_portfolio"

        # Required params
        plan["asof_date"] = params.get("asof_date", date.today()).isoformat() if params.get("asof_date") else date.today().isoformat()
        plan["controller"] = params.get("controller", "baseline")
        plan["feature_version"] = params.get("feature_version", config.feature_version)

        # Portfolio constraints - must be configured
        if config.portfolio_max_gross_notional <= 0:
            if "portfolio_max_gross_notional" not in params:
                missing.append("portfolio_max_gross_notional")
            else:
                plan["portfolio_max_gross_notional"] = params["portfolio_max_gross_notional"]
        else:
            plan["portfolio_max_gross_notional"] = params.get(
                "portfolio_max_gross_notional", config.portfolio_max_gross_notional
            )

        if config.portfolio_max_name_gross_notional <= 0:
            if "portfolio_max_name_gross_notional" not in params:
                missing.append("portfolio_max_name_gross_notional")
            else:
                plan["portfolio_max_name_gross_notional"] = params["portfolio_max_name_gross_notional"]
        else:
            plan["portfolio_max_name_gross_notional"] = params.get(
                "portfolio_max_name_gross_notional", config.portfolio_max_name_gross_notional
            )

        plan["seed"] = params.get("seed", 42)
        plan["model_version"] = params.get("model_version")

    elif intent == Intent.RUN_BACKTEST:
        plan["action"] = "run_backtest"

        # Required params
        if "start_date" not in params and "lookback_months" not in params:
            missing.append("start_date")
        elif "lookback_months" in params:
            from datetime import timedelta
            months = params["lookback_months"]
            plan["start_date"] = (date.today() - timedelta(days=months * 30)).isoformat()
        else:
            plan["start_date"] = params["start_date"].isoformat() if isinstance(params.get("start_date"), date) else params.get("start_date")

        plan["end_date"] = params.get("end_date", date.today())
        if isinstance(plan["end_date"], date):
            plan["end_date"] = plan["end_date"].isoformat()

        plan["controller"] = params.get("controller", "baseline")
        plan["feature_version"] = params.get("feature_version", config.feature_version)
        plan["include_previous_strategies"] = params.get("include_previous_strategies", False)

        # Portfolio constraints
        if config.portfolio_max_gross_notional <= 0:
            if "portfolio_max_gross_notional" not in params:
                missing.append("portfolio_max_gross_notional")
            else:
                plan["portfolio_max_gross_notional"] = params["portfolio_max_gross_notional"]
        else:
            plan["portfolio_max_gross_notional"] = params.get(
                "portfolio_max_gross_notional", config.portfolio_max_gross_notional
            )

        if config.portfolio_max_name_gross_notional <= 0:
            if "portfolio_max_name_gross_notional" not in params:
                missing.append("portfolio_max_name_gross_notional")
            else:
                plan["portfolio_max_name_gross_notional"] = params["portfolio_max_name_gross_notional"]
        else:
            plan["portfolio_max_name_gross_notional"] = params.get(
                "portfolio_max_name_gross_notional", config.portfolio_max_name_gross_notional
            )

        plan["seed"] = params.get("seed", 42)
        plan["model_version"] = params.get("model_version")

    elif intent == Intent.DRAFT_TICKET:
        plan["action"] = "draft_ticket"

        # Check if YAML content provided
        if "yaml_content" in params:
            plan["mode"] = "lint"
            plan["yaml_content"] = params["yaml_content"]
        elif "leg_a" in params and "leg_b" in params:
            plan["mode"] = "create"
            plan["leg_a"] = params["leg_a"]
            plan["leg_b"] = params["leg_b"]
            # Other params may need to be collected
            required_draft_params = [
                "orientation", "max_gross_notional", "max_leg_notional",
                "conviction", "horizon_days", "time_stop_days", "expires_on"
            ]
            for p in required_draft_params:
                if p in params:
                    plan[p] = params[p]
                else:
                    missing.append(p)
        else:
            plan["mode"] = "interactive"
            missing.append("leg_a")
            missing.append("leg_b")

    return plan, missing


def execute_plan(
    plan: Dict[str, Any],
    config: Config,
    console: Console,
) -> Dict[str, Any]:
    # Execute a plan and return the result
    action = plan.get("action")
    result = {"success": False}

    if action == "summarize_strategy":
        result = _execute_summarize(config, console)

    elif action == "generate_ideal_portfolio":
        result = _execute_generate_portfolio(plan, config, console)

    elif action == "run_backtest":
        result = _execute_backtest(plan, config, console)

    elif action == "draft_ticket":
        result = _execute_draft_ticket(plan, config, console)

    return result


def _execute_summarize(config: Config, console: Console) -> Dict[str, Any]:
    # Execute SUMMARIZE intent
    from src.tickets import load_active_tickets
    from src.portfolio_generator import get_latest_portfolio_run

    today = date.today()
    tickets = load_active_tickets(config, today)

    console.print(Panel("[bold]Strategy Summary[/bold]", expand=False))

    # Show tickets
    if not tickets:
        console.print("[yellow]No active tickets found.[/yellow]")
    else:
        table = Table(title=f"Active Tickets ({len(tickets)})")
        table.add_column("Ticket ID", style="cyan")
        table.add_column("Pair", style="green")
        table.add_column("Orientation")
        table.add_column("Conviction", justify="right")
        table.add_column("Max Gross", justify="right")
        table.add_column("Expires")

        for ticket in tickets:
            table.add_row(
                ticket.ticket_id,
                ticket.pair_id,
                ticket.orientation,
                str(ticket.conviction),
                f"${ticket.max_gross_notional:,.0f}",
                str(ticket.expires_on),
            )

        console.print(table)

    # Check for today's portfolio
    latest_run = get_latest_portfolio_run(config, today)

    console.print()
    if latest_run:
        run_id = latest_run.name.replace("run_id=", "")
        console.print(f"[green]Today's ideal portfolio has been generated.[/green]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Path: {latest_run}")

        # Show portfolio summary
        portfolio_path = latest_run / "ideal_portfolio.parquet"
        if portfolio_path.exists():
            import polars as pl
            df = pl.read_parquet(portfolio_path)
            console.print()
            _print_portfolio_summary(df, console)
    else:
        console.print("[yellow]No ideal portfolio generated for today.[/yellow]")
        console.print()
        console.print("[bold]Next step:[/bold]")
        console.print("  > generate todays ideal portfolio")

    return {"success": True, "tickets_count": len(tickets), "has_portfolio": latest_run is not None}


def _execute_generate_portfolio(
    plan: Dict[str, Any],
    config: Config,
    console: Console,
) -> Dict[str, Any]:
    # Execute GENERATE_IDEAL_PORTFOLIO intent
    from src.portfolio_generator import generate_ideal_portfolio

    asof_date = date.fromisoformat(plan["asof_date"])
    controller = plan["controller"]
    feature_version = plan["feature_version"]
    portfolio_max_gross = plan["portfolio_max_gross_notional"]
    portfolio_max_name = plan["portfolio_max_name_gross_notional"]
    seed = plan.get("seed", 42)
    model_version = plan.get("model_version")

    console.print(Panel("[bold]Generating Ideal Portfolio[/bold]", expand=False))
    console.print(f"As-of date: {asof_date}")
    console.print(f"Controller: {controller}")
    console.print(f"Portfolio max gross: ${portfolio_max_gross:,.0f}")
    console.print(f"Portfolio max name: ${portfolio_max_name:,.0f}")
    console.print()

    try:
        result = generate_ideal_portfolio(
            config=config,
            asof_date=asof_date,
            feature_version=feature_version,
            controller_name=controller,
            seed=seed,
            portfolio_max_gross_notional=portfolio_max_gross,
            portfolio_max_name_gross_notional=portfolio_max_name,
            model_version=model_version,
        )

        console.print(f"[green]Portfolio generated successfully![/green]")
        console.print(f"Run ID: {result['run_id']}")
        console.print(f"Path: {result['run_dir']}")
        console.print()

        # Show metrics
        metrics = result["metrics"]
        console.print("[bold]Metrics:[/bold]")
        console.print(f"  Equity Gross: ${metrics['equity_gross']:,.0f}")
        console.print(f"  Equity Net: ${metrics['equity_net']:,.2f}")
        console.print(f"  Cash Notional: ${metrics['cash_notional']:,.0f}")
        console.print(f"  Final Net: ${metrics['final_net']:.2f} (must be 0)")
        console.print(f"  Name caps triggered: {metrics['name_caps_triggered_count']}")
        console.print(f"  Portfolio gross clamped: {metrics['portfolio_gross_clamped']}")
        console.print()

        # Show portfolio
        import polars as pl
        portfolio_path = result["run_dir"] / "ideal_portfolio.parquet"
        if portfolio_path.exists():
            df = pl.read_parquet(portfolio_path)
            _print_portfolio_summary(df, console)

        return {"success": True, "run_id": result["run_id"], "metrics": metrics}

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return {"success": False, "error": str(e)}


def _execute_backtest(
    plan: Dict[str, Any],
    config: Config,
    console: Console,
) -> Dict[str, Any]:
    # Execute RUN_BACKTEST intent
    from src.backtest_runner import run_backtest

    start_date = date.fromisoformat(plan["start_date"])
    end_date = date.fromisoformat(plan["end_date"])
    controller = plan["controller"]
    feature_version = plan["feature_version"]
    portfolio_max_gross = plan["portfolio_max_gross_notional"]
    portfolio_max_name = plan["portfolio_max_name_gross_notional"]
    include_previous = plan.get("include_previous_strategies", False)
    seed = plan.get("seed", 42)
    model_version = plan.get("model_version")

    console.print(Panel("[bold]Running Backtest[/bold]", expand=False))
    console.print(f"Period: {start_date} to {end_date}")
    console.print(f"Controller: {controller}")
    console.print(f"Portfolio max gross: ${portfolio_max_gross:,.0f}")
    console.print(f"Portfolio max name: ${portfolio_max_name:,.0f}")
    console.print(f"Include previous strategies: {include_previous}")
    console.print()

    try:
        result = run_backtest(
            config=config,
            start_date=start_date,
            end_date=end_date,
            feature_version=feature_version,
            controller_name=controller,
            seed=seed,
            portfolio_max_gross_notional=portfolio_max_gross,
            portfolio_max_name_gross_notional=portfolio_max_name,
            include_previous_strategies=include_previous,
            model_version=model_version,
        )

        console.print(f"[green]Backtest completed![/green]")
        console.print(f"Run ID: {result['run_id']}")
        console.print(f"Path: {result['run_dir']}")
        console.print()

        # Show metrics
        metrics = result["metrics"]
        console.print("[bold]Backtest Results:[/bold]")
        console.print(f"  Trading days: {metrics['trading_days']}")
        console.print(f"  Days with data: {metrics['days_with_data']}")
        console.print(f"  Total PnL Gross: ${metrics['total_pnl_gross']:,.2f}")
        console.print(f"  Total PnL Net: ${metrics['total_pnl_net']:,.2f}")
        console.print(f"  Total Costs: ${metrics['total_costs']:,.2f}")
        console.print(f"  Total Turnover: ${metrics['total_turnover']:,.0f}")
        console.print(f"  Avg Daily PnL: ${metrics['avg_daily_pnl_net']:,.2f}")
        console.print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        return {"success": True, "run_id": result["run_id"], "metrics": metrics}

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return {"success": False, "error": str(e)}


def _execute_draft_ticket(
    plan: Dict[str, Any],
    config: Config,
    console: Console,
) -> Dict[str, Any]:
    # Execute DRAFT_TICKET intent
    from src.ticket_drafter import (
        lint_ticket_yaml,
        create_draft_from_params,
        suggest_improvements,
        save_draft_yaml,
    )

    mode = plan.get("mode", "interactive")

    console.print(Panel("[bold]Ticket Drafting[/bold]", expand=False))

    if mode == "lint":
        yaml_content = plan["yaml_content"]
        errors, warnings, parsed_data = lint_ticket_yaml(yaml_content)

        if errors:
            console.print("[red]Errors found:[/red]")
            for err in errors:
                console.print(f"  - {err}")
        else:
            console.print("[green]No errors found.[/green]")

        if warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warn in warnings:
                console.print(f"  - {warn}")

        suggestions = suggest_improvements(parsed_data)
        if suggestions:
            console.print("[cyan]Suggestions:[/cyan]")
            for sug in suggestions:
                console.print(f"  - {sug}")

        # Save as draft
        draft_path = save_draft_yaml(config, yaml_content)
        console.print()
        console.print(f"[green]Draft saved to: {draft_path}[/green]")
        console.print()
        console.print("[bold]Next step:[/bold]")
        console.print(f"  To promote to active: mv {draft_path} {config.tickets_dir}/")

        return {"success": True, "draft_path": str(draft_path), "errors": errors, "warnings": warnings}

    elif mode == "create":
        # Create from params
        try:
            draft_path, yaml_content = create_draft_from_params(
                config=config,
                leg_a=plan["leg_a"],
                leg_b=plan["leg_b"],
                orientation=plan["orientation"],
                max_gross_notional=float(plan["max_gross_notional"]),
                max_leg_notional=float(plan["max_leg_notional"]),
                conviction=int(plan["conviction"]),
                horizon_days=int(plan["horizon_days"]),
                time_stop_days=int(plan["time_stop_days"]),
                expires_on=date.fromisoformat(plan["expires_on"]) if isinstance(plan["expires_on"], str) else plan["expires_on"],
                entry_z=plan.get("entry_z"),
                exit_z=plan.get("exit_z"),
            )

            console.print("[green]Draft ticket created:[/green]")
            console.print()
            console.print(yaml_content)
            console.print()
            console.print(f"[green]Saved to: {draft_path}[/green]")
            console.print()
            console.print("[bold]Next step:[/bold]")
            console.print(f"  To promote to active: mv {draft_path} {config.tickets_dir}/")

            return {"success": True, "draft_path": str(draft_path)}

        except Exception as e:
            console.print(f"[red]Error creating draft: {e}[/red]")
            return {"success": False, "error": str(e)}

    else:
        console.print("[yellow]Interactive ticket creation requires answering questions.[/yellow]")
        console.print("Please specify the legs: e.g., 'draft ticket for JPM and BAC'")
        return {"success": False, "error": "Missing leg symbols"}


def _print_portfolio_summary(df, console: Console):
    # Print a summary of the portfolio
    table = Table(title="Ideal Portfolio")
    table.add_column("Symbol", style="cyan")
    table.add_column("Notional", justify="right")
    table.add_column("Source Tickets")
    table.add_column("Clamps", style="yellow")

    for row in df.sort("symbol").iter_rows(named=True):
        notional = row["notional"]
        color = "green" if notional >= 0 else "red"
        table.add_row(
            row["symbol"],
            f"[{color}]${notional:,.0f}[/{color}]",
            row["source_ticket_ids"] or "-",
            row["clamp_codes"] or "-",
        )

    console.print(table)


def print_help(console: Console):
    # Print help message
    console.print(Panel("[bold]Prompt Console Help[/bold]", expand=False))
    console.print()
    console.print("[bold]Supported commands:[/bold]")
    console.print()
    console.print("  [cyan]SUMMARIZE[/cyan]")
    console.print("    Examples: 'summarize todays strategy', 'whats my strategy', 'status'")
    console.print()
    console.print("  [cyan]GENERATE_IDEAL_PORTFOLIO[/cyan]")
    console.print("    Examples: 'generate todays ideal portfolio', 'build portfolio'")
    console.print()
    console.print("  [cyan]RUN_BACKTEST[/cyan]")
    console.print("    Examples: 'backtest last 6 months', 'run backtest 2025-01-01 to 2025-06-01'")
    console.print()
    console.print("  [cyan]DRAFT_TICKET[/cyan]")
    console.print("    Examples: 'draft ticket for JPM and BAC', 'help me write a ticket'")
    console.print()
    console.print("[bold]Other commands:[/bold]")
    console.print("  'help' - Show this help message")
    console.print("  'quit' or 'exit' - Exit the console")
    console.print()


def print_plan(plan: Dict[str, Any], console: Console):
    # Print the plan for user confirmation
    console.print()
    console.print(Panel("[bold]Plan[/bold]", expand=False))

    for key, value in sorted(plan.items()):
        if key == "timestamp":
            continue
        console.print(f"  {key}: {value}")

    console.print()


def get_param_question(param_name: str) -> str:
    # Get the question to ask for a missing parameter
    questions = {
        "portfolio_max_gross_notional": "What is the portfolio maximum gross notional (in dollars)?",
        "portfolio_max_name_gross_notional": "What is the maximum gross notional per name/symbol (in dollars)?",
        "start_date": "What is the start date for the backtest? (YYYY-MM-DD)",
        "end_date": "What is the end date for the backtest? (YYYY-MM-DD)",
        "leg_a": "What is the first leg symbol? (e.g., JPM)",
        "leg_b": "What is the second leg symbol? (e.g., BAC)",
        "orientation": "What is the orientation? (LONG_SPREAD or SHORT_SPREAD)",
        "max_gross_notional": "What is the ticket max gross notional (in dollars)?",
        "max_leg_notional": "What is the ticket max leg notional (in dollars)?",
        "conviction": "What is the conviction level? (1-5)",
        "horizon_days": "What is the horizon in days?",
        "time_stop_days": "What is the time stop in days?",
        "expires_on": "When does the ticket expire? (YYYY-MM-DD)",
        "flatten_on_breakdown": "Flatten on breakdown? (yes/no)",
    }
    return questions.get(param_name, f"Please provide {param_name}:")


def parse_param_response(param_name: str, response: str) -> Any:
    # Parse user response to a parameter question
    response = response.strip()

    if param_name in ("portfolio_max_gross_notional", "portfolio_max_name_gross_notional",
                      "max_gross_notional", "max_leg_notional"):
        # Parse dollar amount
        response = response.replace("$", "").replace(",", "")
        return float(response)

    elif param_name in ("start_date", "end_date", "expires_on"):
        return date.fromisoformat(response)

    elif param_name in ("conviction", "horizon_days", "time_stop_days"):
        return int(response)

    elif param_name == "flatten_on_breakdown":
        return response.lower() in ("yes", "y", "true", "1")

    elif param_name == "orientation":
        response = response.upper()
        if response in ("LONG", "LONG_SPREAD", "L"):
            return "LONG_SPREAD"
        elif response in ("SHORT", "SHORT_SPREAD", "S"):
            return "SHORT_SPREAD"
        return response

    elif param_name in ("leg_a", "leg_b"):
        return response.upper()

    return response


def run_console(config: Config):
    # Main REPL loop
    console = Console()

    console.print()
    console.print(Panel(
        "[bold]Banks Pair-Trading Console[/bold]\n"
        "Type 'help' for available commands, 'quit' to exit.",
        expand=False
    ))
    console.print()

    session = PromptSession(config=config)

    while True:
        try:
            # Get user input
            if session.awaiting_param:
                prompt = f"[{session.awaiting_param}] > "
            elif session.awaiting_confirmation:
                prompt = "[confirm: yes/no] > "
            else:
                prompt = "> "

            user_input = console.input(prompt)
            user_input = user_input.strip()

            if not user_input:
                continue

            # Handle parameter collection
            if session.awaiting_param:
                try:
                    value = parse_param_response(session.awaiting_param, user_input)
                    session.pending_params[session.awaiting_param] = value
                    session.awaiting_param = None

                    # Rebuild plan with new params
                    plan, missing = build_plan(
                        session.pending_intent,
                        session.pending_params,
                        config,
                    )

                    if missing:
                        session.awaiting_param = missing[0]
                        console.print(get_param_question(session.awaiting_param))
                    else:
                        session.last_plan = plan
                        print_plan(plan, console)
                        console.print("Proceed with this plan? (yes/no)")
                        session.awaiting_confirmation = True
                except Exception as e:
                    console.print(f"[red]Invalid input: {e}. Please try again.[/red]")
                continue

            # Handle plan confirmation
            if session.awaiting_confirmation:
                if user_input.lower() in ("yes", "y"):
                    session.awaiting_confirmation = False
                    result = execute_plan(session.last_plan, config, console)
                    if result.get("success"):
                        session.last_run_id = result.get("run_id")
                    session.pending_intent = None
                    session.pending_params = {}
                    session.last_plan = None
                elif user_input.lower() in ("no", "n"):
                    session.awaiting_confirmation = False
                    session.pending_intent = None
                    session.pending_params = {}
                    session.last_plan = None
                    console.print("[yellow]Plan cancelled.[/yellow]")
                else:
                    console.print("Please answer yes or no.")
                continue

            # Parse new intent
            intent, extracted_params = parse_intent(user_input)

            if intent == Intent.QUIT:
                console.print("Goodbye!")
                break

            if intent == Intent.HELP:
                print_help(console)
                continue

            if intent == Intent.UNKNOWN:
                console.print("[yellow]I don't understand that command.[/yellow]")
                console.print("Type 'help' to see available commands.")
                continue

            # For SUMMARIZE, execute immediately
            if intent == Intent.SUMMARIZE:
                _execute_summarize(config, console)
                continue

            # For other intents, build plan and check for missing params
            session.pending_intent = intent
            session.pending_params = extracted_params

            plan, missing = build_plan(intent, extracted_params, config)

            if missing:
                session.awaiting_param = missing[0]
                console.print(get_param_question(session.awaiting_param))
            else:
                session.last_plan = plan
                print_plan(plan, console)
                console.print("Proceed with this plan? (yes/no)")
                session.awaiting_confirmation = True

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit.[/yellow]")
        except EOFError:
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            session.awaiting_confirmation = False
            session.awaiting_param = None
