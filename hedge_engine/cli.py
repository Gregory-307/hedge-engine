"""
Command-line interface for quick risk assessment.

Usage:
    python -m hedge_engine.cli assess BTC 10 45000 44000 --vol 0.05 --age 60
    python -m hedge_engine.cli assess ETH -5 3000 3100 --vol 0.08 --spread 5

For full options:
    python -m hedge_engine.cli assess --help
"""

from __future__ import annotations

import argparse
import sys

from .assessor import assess_inventory_risk
from .models import HedgeRecommendation, InventoryPosition, MarketConditions


def format_usd(value: float) -> str:
    """Format USD value with sign and commas."""
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def format_pct(value: float) -> str:
    """Format percentage."""
    return f"{value * 100:.0f}%"


def print_assessment(result: HedgeRecommendation, position: InventoryPosition, market: MarketConditions) -> None:
    """Print a human-readable assessment to terminal."""
    # Header with action
    action_colors = {
        "HOLD": "\033[92m",  # Green
        "REDUCE": "\033[93m",  # Yellow
        "HEDGE_PARTIAL": "\033[93m",  # Yellow
        "HEDGE_FULL": "\033[91m",  # Red
        "LIQUIDATE": "\033[91m\033[1m",  # Bold Red
    }
    reset = "\033[0m"
    action_str = result.action.value
    color = action_colors.get(action_str, "")

    print()
    print("=" * 60)
    print(f"  {position.asset} POSITION ASSESSMENT")
    print("=" * 60)

    # Summary box
    print()
    side = result.summary.position_side
    notional = result.summary.notional_usd
    age = result.summary.age_hours
    print(f"  Position:  {side} {abs(position.size):.4g} {position.asset}")
    print(f"  Notional:  ${notional:,.0f}")
    print(f"  Age:       {age:.1f} hours")
    print()

    # Action recommendation
    print(f"  {color}>>> {action_str} <<<{reset}")
    if result.summary.hedge_order:
        print(f"  Order:     {result.summary.hedge_order}")
    print()

    # Risk score breakdown
    print("  RISK SCORE BREAKDOWN")
    print("  " + "-" * 40)
    rb = result.risk_breakdown
    print(f"  Loss severity:     {rb.loss_severity:5.1f} / 35")
    print(f"  Position age:      {rb.position_age:5.1f} / 20")
    print(f"  Volatility:        {rb.volatility_regime:5.1f} / 25")
    print(f"  Size vs liquidity: {rb.size_vs_liquidity:5.1f} / 20")
    print("  " + "-" * 40)
    print(f"  TOTAL:             {rb.total:5.1f} / 100")
    print()

    # P&L scenarios
    print("  P&L SCENARIOS (unhedged)")
    print("  " + "-" * 40)
    pnl = result.pnl_scenarios
    print(f"  -5% move:  {format_usd(pnl.move_down_5pct):>12}")
    print(f"  -2% move:  {format_usd(pnl.move_down_2pct):>12}")
    print(f"  +2% move:  {format_usd(pnl.move_up_2pct):>12}")
    print(f"  +5% move:  {format_usd(pnl.move_up_5pct):>12}")
    print()

    # Worst/best case
    print(f"  Worst case: {format_usd(result.summary.worst_case_loss_usd)}")
    print(f"  Best case:  {format_usd(result.summary.best_case_gain_usd)}")
    print()

    # Hedge cost (if applicable)
    if result.suggested_hedge:
        h = result.suggested_hedge
        print("  HEDGE COST")
        print("  " + "-" * 40)
        print(f"  Instrument:    {h.instrument.value}")
        print(f"  Size:          {h.size:.4g} {position.asset}")
        print(f"  Spread cost:   ${h.spread_cost_usd:.2f}")
        print(f"  Funding (1d):  ${h.funding_cost_1d_usd:.2f}")
        print(f"  Total cost:    ${h.total_cost_usd:.2f} ({h.total_cost_bps:.1f} bps)")
        print()

    # Reasoning
    print("  " + "-" * 40)
    print(f"  {result.reasoning}")
    print()
    print(f"  Re-evaluate in {result.re_evaluate_minutes} minutes")
    print("=" * 60)
    print()


def cmd_assess(args: argparse.Namespace) -> int:
    """Run assessment command."""
    position = InventoryPosition(
        asset=args.asset.upper(),
        size=args.size,
        entry_price=args.entry_price,
        age_minutes=args.age,
        unrealized_pnl=args.pnl,
    )

    market = MarketConditions(
        current_price=args.current_price,
        volatility_1d=args.vol,
        spot_spread_bps=args.spread,
        perp_funding_rate=args.funding,
        bid_depth_usd=args.bid_depth,
        ask_depth_usd=args.ask_depth,
    )

    result = assess_inventory_risk(position, market)
    print_assessment(result, position, market)

    # Return non-zero exit code for high-risk actions
    if result.action.value in ("LIQUIDATE", "HEDGE_FULL"):
        return 1
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hedge-engine",
        description="Inventory Risk Assessment Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # assess command
    assess_parser = subparsers.add_parser(
        "assess",
        help="Assess a position's risk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Assess inventory risk for a position.

Examples:
  # Long 10 BTC, entered at $45k, current price $44k
  assess BTC 10 45000 44000

  # Short 5 ETH with high volatility
  assess ETH -5 3000 3100 --vol 0.08

  # Old position with custom spread
  assess BTC 2 50000 48000 --age 300 --spread 5
        """,
    )
    assess_parser.add_argument("asset", help="Asset symbol (e.g., BTC, ETH)")
    assess_parser.add_argument("size", type=float, help="Position size (+long, -short)")
    assess_parser.add_argument("entry_price", type=float, help="Entry price in USD")
    assess_parser.add_argument("current_price", type=float, help="Current price in USD")
    assess_parser.add_argument("--vol", type=float, default=0.05, help="1-day volatility (default: 0.05)")
    assess_parser.add_argument("--age", type=int, default=60, help="Position age in minutes (default: 60)")
    assess_parser.add_argument("--spread", type=float, default=3.0, help="Spot spread in bps (default: 3.0)")
    assess_parser.add_argument("--funding", type=float, default=0.05, help="Perp funding rate (default: 0.05)")
    assess_parser.add_argument("--bid-depth", type=float, default=5_000_000, help="Bid depth in USD")
    assess_parser.add_argument("--ask-depth", type=float, default=5_000_000, help="Ask depth in USD")
    assess_parser.add_argument("--pnl", type=float, default=0.0, help="Current unrealized P&L")
    assess_parser.set_defaults(func=cmd_assess)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
