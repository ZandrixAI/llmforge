#!/usr/bin/env python3
"""
Script to fetch contributors from GitHub and generate a contributors markdown file.
Run this before building the documentation to update the contributors list.
"""

import urllib.request
import json
import sys

REPO = "ZandrixAI/llmforge"
OUTPUT_FILE = "docs/includes/contributors.md"

def fetch_contributors():
    """Fetch contributors from GitHub API"""
    url = f"https://api.github.com/repos/{REPO}/contributors"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"Error fetching contributors: {e}", file=sys.stderr)
        return []

def generate_contributors_md(contributors):
    """Generate markdown for contributors"""
    md = """---
title: Contributors
---

# Contributors

Thank you to all the amazing people who have contributed to LLMForge!

## Active Contributors

"""
    
    if not contributors:
        md += "*No contributors yet. Be the first to contribute!*\n\n"
        return md
    
    for contributor in contributors:
        login = contributor.get('login', 'Unknown')
        contributions = contributor.get('contributions', 0)
        avatar_url = contributor.get('avatar_url', '')
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
*This list is automatically generated from GitHub.*
"""
    
    return md

def main():
    print("Fetching contributors from GitHub...")
    contributors = fetch_contributors()
    
    # Ensure the includes directory exists
    import os
    os.makedirs("docs/includes", exist_ok=True)
    
    md = generate_contributors_md(contributors)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"Generated {OUTPUT_FILE} with {len(contributors)} contributors")

if __name__ == "__main__":
    main()