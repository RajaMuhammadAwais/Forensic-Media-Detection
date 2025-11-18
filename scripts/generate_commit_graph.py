import os
import sys
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_repo_slug() -> str:
    """
    Determine the owner/repo slug.
    Priority:
      1) GITHUB_REPOSITORY env (set in GitHub Actions)
      2) Parse from `git remote get-url origin` if available
    """
    slug = os.getenv("GITHUB_REPOSITORY")
    if slug:
        return slug

    # Fallback: try to parse from git remote url
    try:
        import subprocess

        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True
        ).strip()
        # Possible formats:
        # - git@github.com:owner/repo.git
        # - https://github.com/owner/repo.git
        if url.startswith("git@github.com:"):
            slug = url.split("git@github.com:")[1]
        elif "github.com/" in url:
            slug = url.split("github.com/")[1]
        else:
            raise ValueError("Not a GitHub remote URL")

        if slug.endswith(".git"):
            slug = slug[:-4]
        return slug
    except Exception:
        print(
            "Unable to detect repository slug. Set GITHUB_REPOSITORY=owner/repo.",
            file=sys.stderr,
        )
        sys.exit(1)


def github_api_request(url: str, params: Optional[Dict] = None) -> requests.Response:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "commit-graph-generator",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def fetch_commits(owner_repo: str, since: datetime) -> List[Dict]:
    """
    Fetch commits from GitHub API since a given datetime (UTC).
    Paginates until we either run out or reach commits older than `since`.
    """
    commits: List[Dict] = []
    page = 1
    base_url = f"https://api.github.com/repos/{owner_repo}/commits"

    while True:
        params = {
            "since": since.isoformat().replace("+00:00", "Z"),
            "per_page": 100,
            "page": page,
        }
        resp = github_api_request(base_url, params=params)
        batch = resp.json()
        if not batch:
            break

        commits.extend(batch)

        # If fewer than 100 returned, likely last page
        if len(batch) < 100:
            break

        page += 1

    return commits


def aggregate_commits_per_day(commits: List[Dict]) -> pd.Series:
    """
    Convert commit list into a daily count time series (UTC).
    """
    if not commits:
        # return an empty 30-day series of zeros to keep the plot stable
        today = datetime.now(timezone.utc).date()
        idx = pd.date_range(end=today, periods=30, freq="D")
        return pd.Series(0, index=idx)

    dates: List[datetime] = []
    for c in commits:
        # 'commit.author.date' is ISO 8601
        try:
            date_str = c["commit"]["author"]["date"]
        except KeyError:
            # fallback to committer date if needed
            date_str = c["commit"]["committer"]["date"]
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        dates.append(dt)

    s = pd.Series(1, index=pd.to_datetime(dates))
    daily = s.resample("D").sum().sort_index()

    # Fill missing dates
    full_idx = pd.date_range(start=daily.index.min().date(), end=datetime.now(timezone.utc).date(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    return daily


def plot_commits(daily_counts: pd.Series, out_path: str):
    """
    Plot a clean, minimal line chart and save to PNG.
    """
    # Minimal, GitHub-friendly style
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4),
            "axes.edgecolor": "#e5e7eb",  # light gray border
            "axes.labelcolor": "#111827",  # near-black text
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.grid": True,
            "grid.color": "#f3f4f6",
            "grid.linestyle": "-",
            "grid.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
        }
    )

    fig, ax = plt.subplots()

    # In case the series covers a long period, restrict to last 365 days for readability
    if len(daily_counts) > 365:
        daily_counts = daily_counts.iloc[-365:]

    # Plot line
    ax.plot(
        daily_counts.index,
        daily_counts.values,
        color="#2563eb",  # blue-600
        linewidth=2.0,
        label="Commits per day",
    )

    # Optional subtle rolling average for smoother trend
    if len(daily_counts) >= 7:
        rolling = daily_counts.rolling(window=7, min_periods=1).mean()
        ax.plot(
            rolling.index,
            rolling.values,
            color="#111827",
            linewidth=1.5,
            alpha=0.5,
            label="7-day avg",
        )

    # Labels and ticks
    ax.set_ylabel("Commits per day")
    ax.set_xlabel("Date")

    # X-axis date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Light y grid only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Legend
    ax.legend(frameon=False, loc="upper left")

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig.savefig(out_path)
    plt.close(fig)


def main():
    owner_repo = get_repo_slug()

    # Look back 1 year for activity; adjust if needed
    since = datetime.now(timezone.utc) - timedelta(days=365)

    commits = fetch_commits(owner_repo, since)
    daily_counts = aggregate_commits_per_day(commits)
    out_path = os.path.join("assets", "graph.png")
    plot_commits(daily_counts, out_path)
    print(f"Generated {out_path} with {int(daily_counts.sum())} total commits in range.")


if __name__ == "__main__":
    main()