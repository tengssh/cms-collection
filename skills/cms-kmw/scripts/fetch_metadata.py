"""
Step 1: Metadata & Readme Retrieval
Autonomous script to fetch repository details and README file from GitHub or Gitee.
"""

import urllib.request
import re
import json
import ssl
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari"
}

def fetch_github_metadata(owner, repo, token=None):
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
    readme_url_fallback = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
    
    # Retrieve token from environment if not supplied
    if not token:
        token = os.getenv("GITHUB_TOKEN")
        
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    metadata = {}
    
    # 1. Fetch Repository Details
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            metadata["name"] = data.get("name")
            metadata["description"] = data.get("description")
            metadata["license"] = data.get("license", {}).get("spdx_id") if data.get("license") else None
            metadata["stars"] = data.get("stargazers_count")
            metadata["updated_at"] = data.get("updated_at")
            metadata["status"] = "success"
    except Exception as e:
        metadata["status"] = f"error api: {e}"
        
    # 2. Fetch README
    readme_content = None
    for url in [readme_url, readme_url_fallback]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                readme_content = response.read().decode('utf-8', errors='ignore')
                break
        except Exception:
            try:
                # Retry without auth header for raw content fallback
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=10) as response:
                    readme_content = response.read().decode('utf-8', errors='ignore')
                    break
            except Exception:
                continue
            
    metadata["readme"] = readme_content if readme_content else "Not Found"
    return metadata

if __name__ == "__main__":
    # Example execution: Fetching mlatom metadata
    owner_name = "dralgroup"
    repo_name = "mlatom"
    
    print(f"Retrieving metadata for {owner_name}/{repo_name}...")
    res = fetch_github_metadata(owner_name, repo_name)
    
    print("\n--- Extracted Metadata ---")
    print(f"Name: {res.get('name')}")
    print(f"Description: {res.get('description')}")
    print(f"License: {res.get('license')}")
    print(f"Updated At: {res.get('updated_at')}")
    print(f"Readme snippet (first 200 chars): {res.get('readme')[:200]}...")
