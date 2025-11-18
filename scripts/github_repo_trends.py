#!/usr/bin/env python3
"""
Generate repository trend graphs using GitHub REST API and save PNGs to repo root.

Outputs:
- code_frequency.png: Weekly additions vs deletions (last ~52 weeks)
- issues_pr_trend.png: Weekly opened Issues vs PRs (last ~52 weeks)

Environment:
- GITHUB_REPOSITORY: owner/repo (provided by GitHub Actions)
- GITHUB_TOKEN or GH_TOKEN: token for authenticated requests to raise rate limits
"""

from __future__ import annotations

import os
import sys
import time
import datetime as dt
from typing import Dict, List, Tuple

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


API_ROOT = "https://api.github.com"


def get_owner_repo() -> Tuple[str, str]:
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo or "/" not in repo:
        print("Error: GITHUB_REPOSITORY must be set to owner/repo", file=sys.stderr)
        sys.exit(2)
    owner, name = repo.split("/", 1)
    return owner, name


def get_session() -> requests.Session:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("TOKEN")
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-trends-generator",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def fetch_code_frequency(session: requests.Session, owner: str, repo: str) -> List[Tuple[int, int, int]]:
    """
    Returns list of tuples: (unix_week_start, additions, deletions)
    Docs: GET /repos/{owner}/{repo}/stats/code_frequency
    This endpoint may return 202 if being generated; we retry a few times.
    """
    url = f"{API_ROOT}/repos/{owner}/{repo}/stats/code_frequency"
    for i in range(10):
        resp = session.get(url, timeout=30)
        if resp.status_code == 202:
            time.sleep(3)
            continue
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return [(int(w[0]), int(w[1]), abs(int(w[2]))) for w in data]
        break
    return []


def daterange_weeks(end_date: dt.date, weeks: int) -> List[Tuple[dt.date, dt.date]]:
    """
    Return list of (start, end) week windows ending on end_date (inclusive), most-recent first.
    Each window spans 7 days: [start, end].
    """
    windows = []
    cur_end = end_date
    for _ in range(weeks):
        start = cur_end - dt.timedelta(days=6)
        windows.append((start, cur_end))
        cur_end = start - dt.timedelta(days=1)
    windows.reverse()
    return windows


def count_items_created(session: requests.Session, owner: str, repo: str, start: dt.date, end: dt.date, item_type: str) -> int:
    """
    Count number of items created in the window for item_type in {"issue","pr"} using Search API.
    Uses created date range and repo qualifier.
    """
    # Search issues covers both issues and PRs; distinguish by 'is:issue' or 'is:pr'
    q = f"repo:{owner}/{repo} is:{item_type} created:{start}..{end}"
    url = f"{API_ROOT}/search/issues"
    total = 0
    page = 1
    per_page = 100
    while True:
        params = {"q": q, "per_page": per_page, "page": page}
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            time.sleep(30)
            continue
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        total += len(items)
        if len(items) < per_page:
            break
        page += 1
    return total


def generate_code_frequency_png(weekly: List[Tuple[int, int, int]], out_path: str) -> None:
    """
    weekly: list of (unix_week_start, additions, deletions)
    """
    if not weekly:
        # create an empty but valid plot
        plt.figure(figsize=(10, 4), dpi=150)
        plt.title("Code Frequency (Additions vs Deletions)")
        plt.xlabel("Week")
        plt.ylabel("Lines changed")
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    # Convert to datetime and restrict to last ~52 weeks
    points = [(dt.datetime.utcfromtimestamp(w).date(), a, d) for (w, a, d) in weekly]
    if len(points) > 60:
        points = points[-60:]

    weeks = [p[0] for p in points]
    additions = [p[1] for p in points]
    deletions = [p[2] for p in points]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.plot(weeks, additions, color="#10b981", linewidth=2, label="Additions")
    ax.plot(weeks, deletions, color="#ef4444", linewidth=2, label="Deletions")

    ax.set_title("Code Frequency (Weekly Additions vs Deletions)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Lines changed")

    fig.autofmt_xdate(rotation=30)
    ax.margins(x=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_issues_pr_trend_png(session: requests.Session, owner: str, repo: str, weeks_back: int, out_path: str) -> None:
    """
    Build per-week trend for opened issues and PRs.
    """
    end = dt.datetime.utcnow().date()
    windows = daterange_weeks(end, weeks_back)

    week_labels: List[dt.date] = []
    issues_counts: List[int] = []
    prs_counts: List[int] = []

    for start, endw in windows:
        week_labels.append(endw)
        issues_counts.append(count_items_created(session, owner, repo, start, endw, "issue"))
        prs_counts.append(count_items_created(session, owner, repo, start, endw, "pr"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.plot(week_labels, issues_counts, color="#3b82f6", linewidth=2, label="Issues opened")
    ax.plot(week_labels, prs_counts, color="#8b5cf6", linewidth=2, label="PRs opened")

    ax.set_title("Issues & Pull Requests Trend (weekly opened)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Week (end)")
    ax.set_ylabel("Count")

    fig.autofmt_xdate(rotation=30)
    ax.margins(x=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    owner, repo = get_owner_repo()
    session = get_session()

    # Code frequency (additions vs deletions)
    weekly_cf = fetch_code_frequency(session, owner, repo)
    generate_code_frequency_png(weekly_cf, "code_frequency.png")

    # Issues & PR trend (last 26 weeks)
    generate_issues_pr_trend_png(session, owner, repo, weeks_back=26, out_path="issues_pr_trend.png")

    print("Generated: code_frequency.png, issues_pr_trend.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())