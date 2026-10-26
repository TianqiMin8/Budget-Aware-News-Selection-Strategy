"""Command-line entry point for checking news data and comparing budgeted selectors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from newsalpha.ingestion import load_news
from newsalpha.selection import METHODS, select_news


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newsalpha")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-news", help="validate a news CSV")
    validate.add_argument("--input", required=True, type=Path)

    select = commands.add_parser("select", help="select candidate news under a token budget")
    select.add_argument("--input", required=True, type=Path)
    select.add_argument("--ticker", required=True)
    select.add_argument("--candidate-start", required=True, help="Inclusive candidate-window start timestamp")
    select.add_argument("--as-of", required=True, help="Timestamp with timezone, e.g. 2024-01-03T09:30:00-05:00")
    select.add_argument("--budget", required=True, type=int)
    select.add_argument("--method", choices=METHODS, required=True)
    select.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        news = load_news(args.input)
        if args.command == "validate-news":
            print(f"Valid news dataset: {len(news)} articles")
            return 0

        start = pd.to_datetime(args.candidate_start, utc=True, errors="coerce")
        cutoff = pd.to_datetime(args.as_of, utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(cutoff) or start >= cutoff:
            raise ValueError("candidate-start and as-of must be valid timestamps with candidate-start before as-of")
        ticker_news = news.loc[news["ticker"].str.casefold() == args.ticker.casefold()]
        historical = ticker_news.loc[ticker_news["published_at"] < start]
        candidates = ticker_news.loc[
            (ticker_news["published_at"] >= start) & (ticker_news["published_at"] <= cutoff)
        ].copy()
        if candidates.empty:
            raise ValueError("No candidate articles fall within the requested ticker and time window")
        selected = select_news(
            candidates,
            args.method,
            args.budget,
            historical_articles=historical,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.output, index=False)
        tokens_used = int(selected["token_count_used"].sum())
        print(
            f"Selected {len(selected)} articles for {args.ticker}; "
            f"{tokens_used}/{args.budget} tokens; wrote {args.output}"
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
