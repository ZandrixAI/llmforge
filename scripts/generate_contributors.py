#!/usr/bin/env python3
"""
Script to fetch contributors from GitHub and generate a contributors markdown file.
Only shows contributors who made commits AFTER the fork (Oct 2025+).
"""

import urllib.request
import json
import sys
from datetime import datetime, timezone

REPO = "ZandrixAI/llmforge"
OUTPUT_FILE = "docs/includes/contributors.md"

# Fork date - only count contributors after this date (with timezone)
# Setting to April 2026 when LLMForge fork was created
FORK_DATE = datetime(2026, 4, 1, tzinfo=timezone.utc)


def fetch_contributors():
    """Fetch contributors from GitHub API - filtered to commits after fork date"""
    url = f"https://api.github.com/repos/{REPO}/commits?per_page=100&sha=main"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            commits = json.loads(response.read().decode())

        # Extract unique authors from commits, filtering by date
        author_ids = {}
        for commit in commits:
            commit_date = None
            if "commit" in commit and "author" in commit["commit"]:
                date_str = commit["commit"]["author"]["date"]
                # Parse the ISO format date with timezone
                commit_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

            # Only count commits after fork date
            if commit_date and commit_date >= FORK_DATE:
                if "author" in commit and commit["author"] is not None:
                    author_id = commit["author"]["id"]
                    if author_id not in author_ids:
                        author_ids[author_id] = {
                            "login": commit["author"]["login"],
                            "avatar_url": commit["author"]["avatar_url"],
                            "contributions": 0,
                        }
                    author_ids[author_id]["contributions"] += 1

        # Convert to list sorted by contributions
        contributors = list(author_ids.values())
        contributors.sort(key=lambda x: x["contributions"], reverse=True)
        return contributors

    except Exception as e:
        print(f"Error fetching contributors: {e}", file=sys.stderr)
        return []


def generate_contributors_md(contributors):
    """Generate markdown for contributors"""
    md = """---
title: Contributors
---

# Contributors

LLMForge is a fork of [mlx-lm](https://github.com/ml-explore/mlx-lm). Below are the contributors who have contributed specifically to LLMForge since the fork.

## Core Team

<div class="contributor-card">
<a href="https://github.com/DawoodTouseef" target="_blank">
<img src="https://avatars.githubusercontent.com/u/97373719?v=4&s=64" alt="DawoodTouseef" class="contributor-avatar">
<span class="contributor-name">DawoodTouseef</span>
<span class="contributor-count">Lead Developer</span>
</a>
</div>

## LLMForge Contributors

"""

    if not contributors:
        md += "*No contributors yet. Be the first to contribute!*\n\n"
        return md

    for contributor in contributors:
        login = contributor.get("login", "Unknown")
        contributions = contributor.get("contributions", 0)
        avatar_url = contributor.get("avatar_url", "")
        profile_url = f"https://github.com/{login}"

        md += f"""<div class="contributor-card">
<a href="{profile_url}" target="_blank">
<img src="{avatar_url}&s=64" alt="{login}" class="contributor-avatar">
<span class="contributor-name">{login}</span>
<span class="contributor-count">{contributions} contributions</span>
</a>
</div>

"""

    md += """## How to Get Listed

To be automatically listed here:
1. Fork the repository
2. Make your contributions
3. Submit a Pull Request

Once your PR is merged, you will appear in this list!

---
*This list only shows contributors who contributed after October 2025.*
"""

    return md


def main():
    print("Fetching LLMForge-specific contributors (post-fork)...")
    contributors = fetch_contributors()

    # Ensure the includes directory exists
    import os

    os.makedirs("docs/includes", exist_ok=True)

    md = generate_contributors_md(contributors)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Generated {OUTPUT_FILE} with {len(contributors)} contributors")


if __name__ == "__main__":
    main()
