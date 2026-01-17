#!/usr/bin/env python3
"""
Run example scenarios through the risk assessment engine.

Usage:
    python examples/run_examples.py           # Use local library
    python examples/run_examples.py --api     # Use running API server
"""

import json
import sys
from pathlib import Path

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_local(scenario: dict) -> dict:
    """Run scenario using local library."""
    from hedge_engine.assessor import assess_inventory_risk
    from hedge_engine.models import InventoryPosition, MarketConditions

    position = InventoryPosition(**scenario["position"])
    market = MarketConditions(**scenario["market"])
    result = assess_inventory_risk(position, market)

    return {
        "action": result.action.value,
        "hedge_pct": result.hedge_pct,
        "risk_score": round(result.risk_score, 1),
        "reasoning": result.reasoning,
        "pnl_scenarios": {
            "move_down_5pct": result.pnl_scenarios.move_down_5pct,
            "move_up_5pct": result.pnl_scenarios.move_up_5pct,
        },
    }


def run_api(scenario: dict, base_url: str = "http://localhost:8000") -> dict:
    """Run scenario using API server."""
    import httpx

    response = httpx.post(
        f"{base_url}/assess",
        json={"position": scenario["position"], "market": scenario["market"]},
    )
    response.raise_for_status()
    return response.json()


def main():
    use_api = "--api" in sys.argv
    examples_dir = Path(__file__).parent

    print("=" * 70)
    print("INVENTORY RISK ASSESSMENT - EXAMPLE SCENARIOS")
    print("=" * 70)
    print(f"Mode: {'API' if use_api else 'Local library'}\n")

    for json_file in sorted(examples_dir.glob("*.json")):
        with open(json_file) as f:
            scenario = json.load(f)

        print(f"\n{'-' * 70}")
        print(f"File: {json_file.name}")
        print(f"   {scenario.get('description', 'No description')}")
        print(f"   Expected: {scenario.get('expected_action', 'N/A')}")
        print(f"{'-' * 70}")

        try:
            if use_api:
                result = run_api(scenario)
            else:
                result = run_local(scenario)

            print(f"   Action:     {result['action']}")
            print(f"   Hedge %:    {result['hedge_pct'] * 100:.0f}%")
            print(f"   Risk Score: {result['risk_score']}/100")
            print(f"   Reasoning:  {result['reasoning'][:80]}...")

            if "pnl_scenarios" in result:
                pnl = result["pnl_scenarios"]
                down = pnl.get("move_down_5pct", 0)
                up = pnl.get("move_up_5pct", 0)
                print(f"   P&L (-5%):  ${down:,.0f}")
                print(f"   P&L (+5%):  ${up:,.0f}")

        except Exception as e:
            print(f"   ERROR: {e}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
