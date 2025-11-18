#!/usr/bin/env python3
"""
Generate a daily commits line graph (graph.png) for the current GitHub repository.

- Fetches commit history via GitHub REST API (v3)
- Aggregates commits per calendar day (UTC)
- Produces a clean, modern line plot saved to ./graph.png

Environment variables:
- GITHUB_REPOSITORY: "owner/repo" (auto-provided in GitHub Actions)
- GITHUB_TOKEN: GitHub token for authenticated requests (recommended in CI)
Optional CLI args:
- --days N (default: 365) window of days to include

Usage:
  python scripts/github_commits_graph.py --days 365
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Tuple
import time

import requests
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt


API_ROOT = "https://api.github.com"


def _isoformat(dt_obj: dt.datetime) -> str:
    # GitHub API expects ISO8601 with Z (UTC)
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_only(iso_timestamp: str) -> dt.date:
    # Handles e.g. "2025-01-31T12:34:56Z"
    # Some commits may have timezone offsets, but GitHub returns Z usually.
    try:
        # Fast path for Zulu time
        return dt.datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        # Fallback: let fromisoformat handle offsets, then convert to date
        # Replace 'Z' with '+00:00' for fromisoformat compatibility
        iso = iso_timestamp.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(iso).date()


def _paginate_commits(
    owner: str,
    repo: str,
    token: str | None,
    since_iso: str,
    until_iso: str,
    per_page: int = 100,
) -> Iterable[Dict]:
    """
    Generator over commits in the time window [since, until).
    Paginates using Link headers.
    """
    url = f"{API_ROOT}/repos/{owner}/{repo}/commits"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    params = {
        "since": since_iso,
        "until": until_iso,
        "per_page": per_page,
    }
    session = requests.Session()

    while True:
        resp = session.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            # Simple backoff for secondary rate limits
            time.sleep(30)
            continue
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            break
        if not items:
            break
        for it in items:
            yield it

        # Parse pagination from Link header
        link = resp.headers.get("Link", "")
        next_url = None
        if link:
            parts = [p.strip() for p in link.split(",")]
            for p in parts:
                if 'rel="next"' in p:
                    # <https://...>; rel="next"
                    lt = p.split(";")[0].strip()
                    if lt.startswith("<") and lt.endswith(">"):
                        next_url = lt[1:-1]
                        break
        if next_url:
            # When following next link, clear params so URL is used as-is
            url = next_url
            params = {}
        else:
            break


def fetch_commits_per_day(owner: str, repo: str, token: str | None, days: int) -> OrderedDict:
    """
    Return an OrderedDict mapping date -> commit_count for the last `days` days (inclusive today).
    Missing days will be present with 0 count to make the line continuous.
    """
    utc_today = dt.datetime.utcnow().date()
    start_date = utc_today - dt.timedelta(days=days - 1)
    since_iso = _isoformat(dt.datetime.combine(start_date, dt.time.min))
    until_iso = _isoformat(dt.datetime.combine(utc_today + dt.timedelta(days=1), dt.time.min))

    counts = Counter()

    for commit in _paginate_commits(owner, repo, token, since_iso, until_iso):
        # Prefer author date; fall back to committer date
        commit_data = commit.get("commit", {})
        author_info = commit_data.get("author") or {}
        committer_info = commit_data.get("committer") or {}
        timestamp = author_info.get("date") or committer_info.get("date")
        if not timestamp:
            continue
        day = _date_only(timestamp)
        # Only count if in our window (should be, but be safe)
        if start_date <= day <= utc_today:
            counts[day] += 1

    # Fill all days with 0 where missing
    ordered: "OrderedDict[dt.date, int]" = OrderedDict()
    d = start_date
    while d <= utc_today:
        ordered[d] = counts.get(d, 0)
        d += dt.timedelta(days=1)

    return ordered


def plot_line_graph(data_by_day: "OrderedDict[dt.date, int]", output_path: str) -> None:
    """
    Create and save a clean, modern line graph to output_path.
    X-axis: dates; Y-axis: commits per day
    """
    dates = list(data_by_day.keys())
    counts = list(data_by_day.values())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(dates, counts, color="#2b6cb0", linewidth=2, marker=None)

    ax.set_title("Commits per day", fontsize=14, fontweight="bold", color="#222222")
    ax.set_xlabel("Date (UTC)", fontsize=11)
    ax.set_ylabel("Number of commits", fontsize=11)

    # Improve date formatting
    fig.autofmt_xdate(rotation=30)
    ax.margins(x=0)

    # Light ticks and grid
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout and save
    plt.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate commits-per-day line graph for this repository.")
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""), help='owner/repo (default: $GITHUB_REPOSITORY)')
    parser.add_argument("--days", type=int, default=365, help="Number of days to include (default: 365)")
    parser.add_argument("--output", default="graph.png", help="Output PNG path (default: graph.png)")
    args = parser.parse_args(argv)

    if not args.repo or "/" not in args.repo:
        print("Error: --repo must be provided as owner/repo (or set GITHUB_REPOSITORY).", file=sys.stderr)
        return 2

    owner, repo = args.repo.split("/", 1)
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

    data = fetch_commits_per_day(owner, repo, token, days=args.days)
    plot_line_graph(data, args.output)
    print(f"Wrote {args.output} with {len(data)} days.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))