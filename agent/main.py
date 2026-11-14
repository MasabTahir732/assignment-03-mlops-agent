"""CLI: no server, paid model, or API key is required for the rules planner."""

import argparse
import json
import logging
from uuid import uuid4

from dotenv import load_dotenv

from agent.orchestrator import TOOLS, ask, execute, tool_schemas
from agent.tools.common import ROOT


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Evidence-based MLOps agent")
    parser.add_argument("prompt", nargs="?", help="Natural-language request")
    parser.add_argument("--mode", choices=["rules", "llm"], default="rules")
    parser.add_argument(
        "--confirm", action="store_true", help="Explicitly authorize requested write"
    )
    parser.add_argument("--tool", choices=sorted(TOOLS))
    parser.add_argument("--args", default="{}", help="JSON structured tool arguments")
    parser.add_argument("--list-tools", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.list_tools:
        output = tool_schemas()
    elif args.tool:
        try:
            arguments = json.loads(args.args)
            if not isinstance(arguments, dict):
                raise ValueError("Expected a JSON object")
        except ValueError as exc:
            parser.error(str(exc))
        result = execute(args.tool, arguments, uuid4().hex, args.confirm)
        output = result.model_dump()
    elif args.prompt:
        output = ask(args.prompt, args.mode, args.confirm).model_dump()
    else:
        parser.error("Supply a prompt, --tool, or --list-tools")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
