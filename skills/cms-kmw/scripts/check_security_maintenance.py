"""
Step 3: Security & Maintenance Screening
Validates resource safety, URL integrity, license types, and update frequency.
"""

from datetime import datetime
import urllib.parse
import urllib.request

# List of trusted source domains for computational materials science resources
TRUSTED_DOMAINS = [
    "github.com", "gitlab.com", "gitee.com", "huggingface.co", 
    "zenodo.org", "figshare.com", "gitlab.in2p3.fr", "tritondft.com", 
    "elagente.ca", "optimat.chat", "materialscloud.org", "materialsproject.org",
    "openkim.org", "oqmd.org", "nomad-lab.eu", "crystallography.net",
    "nist.gov", "nims.go.jp", "ccdc.cam.ac.uk", "calypso.cn", "aiida.net",
    "pyiron.org"
]

# Standard open source software license identifiers
APPROVED_OSS_LICENSES = {
    "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "GPL-3.0", 
    "GPL-2.0", "LGPL-3.0", "LGPL-2.1", "AGPL-3.0", "Unlicense", "CC0-1.0"
}

def verify_url_active(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari"
    }
    try:
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status in [200, 301, 302, 303, 307, 308]
    except Exception:
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status in [200, 301, 302, 303, 307, 308]
        except Exception:
            return False

def check_security_and_maintenance(name, url, license_id, updated_at_str, is_archived=False):
    report = {
        "resource": name,
        "url_status": "Passed",
        "license_status": "Passed",
        "maintenance_status": "Passed",
        "warnings": [],
        "verdict": "Secure & Active"
    }

    # 1. URL Domain Verification & Live Health Check
    try:
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        # Match base domains or trusted domains
        is_trusted = False
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                is_trusted = True
                break
                
        if not is_trusted:
            report["url_status"] = "Flagged"
            report["warnings"].append(f"Domain '{domain}' is not in the trusted domain whitelist.")
            
        # Perform live check
        if not verify_url_active(url):
            report["url_status"] = "Flagged"
            report["warnings"].append(f"URL live check failed: '{url}' is unreachable or returned error status.")
            
    except Exception as e:
        report["url_status"] = "Failed"
        report["warnings"].append(f"URL parsing/validation failed: {e}")

    # 2. License Screening
    if not license_id:
        report["license_status"] = "Warning"
        report["warnings"].append("No license declared (possible copyright restrictions).")
    elif license_id not in APPROVED_OSS_LICENSES:
        report["license_status"] = "Notice"
        report["warnings"].append(f"Custom/Non-standard license: {license_id}")

    # 3. Maintenance/Activity Screening
    if is_archived:
        report["maintenance_status"] = "Flagged"
        report["warnings"].append("Repository is archived/read-only by owner.")
        
    if updated_at_str:
        try:
            # Parse ISO format date (e.g. 2026-06-01T12:00:00Z)
            # Handle possible trailing Z or offsets
            clean_date = updated_at_str.replace("Z", "")
            updated_date = datetime.fromisoformat(clean_date)
            # Set target reference date (representing mock current time)
            ref_date = datetime.now()
            delta = ref_date - updated_date
            
            # Flag if unmaintained for more than 730 days (2 years)
            if delta.days > 730:
                report["maintenance_status"] = "Flagged"
                report["warnings"].append(f"Inactivity: Last pushed/updated {delta.days} days ago.")
        except Exception as e:
            report["warnings"].append(f"Could not parse update date: {e}")

    # 4. Overall Verdict
    if any(status == "Flagged" for status in [report["url_status"], report["maintenance_status"]]):
        report["verdict"] = "Rejected / Backlog candidate"
    elif any(status == "Warning" for status in [report["license_status"], report["url_status"]]):
        report["verdict"] = "Accept with Revisions (Check details)"
    else:
        report["verdict"] = "Accepted"

    return report

if __name__ == "__main__":
    # Test cases
    cases = [
        {
            "name": "active_tool",
            "url": "https://github.com/dralgroup/mlatom",
            "license": "MIT",
            "updated": "2026-06-01T00:00:00Z",
            "archived": False
        },
        {
            "name": "abandoned_unlicensed",
            "url": "https://github.com/unknown/oldproject",
            "license": None,
            "updated": "2021-01-01T00:00:00Z",
            "archived": True
        },
        {
            "name": "unknown_domain_site",
            "url": "https://suspiciousdomain.example/project",
            "license": "Apache-2.0",
            "updated": "2026-05-15T00:00:00Z",
            "archived": False
        }
    ]

    for case in cases:
        print("=" * 60)
        res = check_security_and_maintenance(
            case["name"], case["url"], case["license"], case["updated"], case["archived"]
        )
        print(f"Name: {res['resource']}")
        print(f"Verdict: {res['verdict']}")
        if res["warnings"]:
            print("Warnings/Flags:")
            for w in res["warnings"]:
                print(f"  - {w}")
