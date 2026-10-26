# NewsAlpha research scaffold

This repository is a runnable starting point for comparing financial-news selection strategies under a fixed token budget. It intentionally does not include licensed news, market data, an LLM provider, or a claim that the proposed method is already validated.

## What each file does

- `.gitignore` keeps local environments, credentials, downloaded datasets, and generated outputs out of version control.
- `.env.example` lists the environment variables an API-backed integration may need; copy it to `.env` locally.
- `pyproject.toml` defines the package, Python dependencies, test settings, and `newsalpha` CLI.
- `configs/default.json` contains experiment defaults.
- `src/newsalpha/ingestion.py` loads and validates the news CSV, filters news as of a cutoff, and parses article embeddings.
- `src/newsalpha/selection.py` implements recency, relevance, MMR, and a novelty/coverage greedy selector subject to a token cap.
- `src/newsalpha/events.py` defines the frozen-LLM event extraction contract and its auditable result format.
- `src/newsalpha/features.py` turns extracted events and market features into one row per ticker/date.
- `src/newsalpha/model.py` fits and predicts with a Ridge model.
- `src/newsalpha/evaluation.py` provides chronological splitting, daily Rank IC, and date-level bootstrap intervals.
- `src/newsalpha/cli.py` exposes validation and news-selection commands.
- `tests/` contains focused unit tests for the core utilities.
- `data/raw/`, `data/processed/`, and `outputs/` are local data/output directories; their contents are ignored by Git.

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

## News CSV format

Required columns:

| Column | Meaning |
|---|---|
| `article_id` | Stable, unique article identifier |
| `ticker` | Company identifier used for filtering |
| `published_at` | Publication timestamp, parseable by pandas |
| `title` | Article headline |
| `text` | Original article text or a consistently sourced fallback |
| `url` | Source URL |

Optional columns:

| Column | Meaning |
|---|---|
| `collected_at` | Collection timestamp for provenance and update checks |
| `token_count` | Token count from the selected tokenizer; otherwise a rough estimate is used |
| `relevance_score` | Query/company relevance score, required for the relevance and MMR/proposed methods |
| `embedding` | JSON array of floats, required for MMR and novelty/coverage |

Keep the source and collection metadata, respect source terms, and document whether the text is full article text or headline-only. `published_at` is the primary as-of filter; decide and document a market-session cutoff before running experiments so future information cannot enter a prediction.

## CLI quick start

Validate a collected dataset:

```powershell
newsalpha validate-news --input data/raw/news.csv
```

Select articles for one ticker and prediction cutoff. The candidate window and the earlier history used to measure novelty are separate:

```powershell
newsalpha select --input data/raw/news.csv --ticker AAPL --candidate-start 2024-01-02T09:30:00-05:00 --as-of 2024-01-03T09:30:00-05:00 --budget 1200 --method recency --output outputs/selected.csv
```

Available methods: `recency`, `relevance`, `mmr`, and `novelty_coverage`. Similarity-based methods require valid, same-length embedding vectors; `relevance`, `mmr`, and `novelty_coverage` require relevance scores. The novelty/coverage score uses earlier articles before `candidate-start` as historical context, then balances historical dissimilarity, diversity within the selected set, freshness within the candidate window, and relevance with equal weights. Treat those weights as an initial experiment setting, not a validated optimum. Token budget is enforced against each article's `token_count`, or a rough character-based estimate when it is absent.

## Research workflow (implementation milestones)

1. Collect and document sources; validate ticker mapping, duplicate IDs, timestamps, text availability, and update handling.
2. Freeze the candidate-news set and session cutoff for each ticker/date. Compute embeddings and relevance scores without using future returns.
3. Run all selectors on identical candidates and token budgets. Save selected IDs, scores, and token totals for reproducibility.
4. Implement a provider-specific `EventExtractor` in `events.py`; preserve source article IDs in each event and record latency/token costs.
5. Build event features with `features.py`, join only market data available by the prediction cutoff, and fit Ridge on chronological training dates.
6. Evaluate held-out daily Rank IC and date-level confidence intervals. Add ablations and a market-only baseline before interpreting selection effects.
7. Treat assumed transaction-cost-adjusted returns as a secondary analysis, not a live-trading result.

The CLI currently handles news validation and selector comparisons only. LLM extraction, feature assembly from real provider outputs, chronological model orchestration, caching/batching, and return-cost analysis are extension points rather than implemented integrations.
