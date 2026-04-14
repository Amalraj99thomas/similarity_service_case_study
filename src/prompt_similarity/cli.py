#!/usr/bin/env python3
"""Command-line interface for the Prompt Similarity Service.

Usage::

    python -m prompt_similarity.cli <command> [options]

The server must be running::

    uvicorn prompt_similarity.app:app --reload
"""

import argparse
import json
import sys

import httpx

BASE = "http://localhost:8000"


def pretty(data) -> None:
    """Print JSON data with indentation."""
    print(json.dumps(data, indent=2))


def cmd_generate(args):
    """Generate embeddings from a JSON file or re-embed all existing prompts."""
    if args.file:
        with open(args.file) as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            print("Error: JSON file must contain a list of prompt objects.", file=sys.stderr)
            sys.exit(1)
        r = httpx.post(f"{BASE}/api/embeddings/generate", json=prompts, timeout=120)
    else:
        r = httpx.post(f"{BASE}/api/embeddings/generate", params={"regenerate_all": True}, timeout=120)
    r.raise_for_status()
    pretty(r.json())


def cmd_search(args):
    """Run semantic search with a free-text query."""
    r = httpx.post(
        f"{BASE}/api/search/semantic",
        json={"query": args.query, "limit": args.limit, "threshold": args.threshold},
        timeout=30,
    )
    r.raise_for_status()
    pretty(r.json())


def cmd_similar(args):
    """Find prompts similar to a given prompt ID."""
    r = httpx.get(
        f"{BASE}/api/prompts/{args.prompt_id}/similar",
        params={"limit": args.limit, "threshold": args.threshold},
        timeout=30,
    )
    r.raise_for_status()
    pretty(r.json())


def cmd_duplicates(args):
    """Detect duplicate prompt clusters."""
    r = httpx.get(
        f"{BASE}/api/analysis/duplicates",
        params={"threshold": args.threshold},
        timeout=60,
    )
    r.raise_for_status()
    pretty(r.json())


def cmd_health(_args):
    """Check service health."""
    r = httpx.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    pretty(r.json())


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="prompt-similarity",
        description="Prompt Similarity Service — CLI",
    )
    parser.add_argument("--base", default=BASE, help="Base URL of the service")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate embeddings. Pass --file to upload new prompts.")
    gen.add_argument(
        "--file", metavar="prompts.json",
        help="Path to a JSON file containing a list of prompt objects. "
             "Omit to re-embed all prompts already in the DB.",
    )
    gen.set_defaults(func=cmd_generate)

    # search
    srch = sub.add_parser("search", help="Semantic search by free-text query")
    srch.add_argument("query", help="Natural-language search query")
    srch.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    srch.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum similarity score 0–1 (default: 0.0)",
    )
    srch.set_defaults(func=cmd_search)

    # similar
    sim = sub.add_parser("similar", help="Find prompts similar to a given prompt ID")
    sim.add_argument("prompt_id", help="Source prompt ID")
    sim.add_argument("--limit", type=int, default=5)
    sim.add_argument("--threshold", type=float, default=0.7)
    sim.set_defaults(func=cmd_similar)

    # duplicates
    dup = sub.add_parser("duplicates", help="Find clusters of near-duplicate prompts")
    dup.add_argument(
        "--threshold", type=float, default=0.9,
        help="Similarity threshold for duplicate detection (default: 0.9)",
    )
    dup.set_defaults(func=cmd_duplicates)

    # health
    hlth = sub.add_parser("health", help="Check service health")
    hlth.set_defaults(func=cmd_health)

    return parser


def main():
    """CLI entry point."""
    global BASE

    parser = build_parser()
    args = parser.parse_args()

    # Allow --base override globally
    BASE = args.base

    try:
        args.func(args)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Could not connect to {BASE}. Is the server running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
