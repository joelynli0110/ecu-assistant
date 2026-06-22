"""Command-line interface for the ECU assistant."""

from __future__ import annotations

import argparse
import json

from ecu_assistant.agent.graph import ECUEngineeringAgent


def main() -> None:
    """Parse a question and print one structured response."""

    parser = argparse.ArgumentParser(description="Ask the ME ECU engineering assistant.")
    parser.add_argument("query", nargs="+", help="Engineering question")
    args = parser.parse_args()
    response = ECUEngineeringAgent().invoke(" ".join(args.query))
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

