"""
Step 5: Cataloging & Description Styling
Formats descriptions to adhere strictly to the repository style guide, 
detects commercial properties, and outputs ready-to-use markdown table rows.
"""

import re

# Redundant prefixes to strip from descriptions
REDUNDANT_PREFIX_PATTERNS = [
    r"^a\s+python\s+package\s+for\s+",
    r"^a\s+python\s+library\s+for\s+",
    r"^code\s+for\s+",
    r"^official\s+implementation\s+of\s+",
    r"^a\s+collection\s+of\s+",
    r"^a\s+library\s+for\s+",
    r"^software\s+for\s+",
    r"^an\s+open-source\s+tool\s+for\s+",
    r"^a\s+framework\s+for\s+"
]

# Case-insensitive commercial indicators
COMMERCIAL_KEYWORDS = [
    r"\bcommercial\b", r"\bproprietary\b", r"\blicense\s+required\b", 
    r"\bsubscription\b", r"\benterprise\b"
]

# Scientific acronyms and proper nouns that must preserve their exact casing
PRESERVE_CASING_TERMS = {
    "SE(3)", "SE3", "SO(3)", "SO3", "DFT", "MD", "ML", "AI", "GNN", "MLIP", 
    "XANES", "QM/MM", "PIMD", "PDE", "GPU", "PINN", "VASP", "LAMMPS", "CP2K", 
    "QE", "API", "DFT-based", "ML-based"
}

def clean_description(description, is_commercial_override=False):
    # 1. Clean whitespace
    desc = description.strip()
    
    # 2. Strip redundant prefixes
    for pattern in REDUNDANT_PREFIX_PATTERNS:
        desc = re.sub(pattern, "", desc, flags=re.IGNORECASE)
        
    # 3. Strip trailing license mentions like (MIT License), (no license), etc.
    desc = re.sub(r"\s*\([^)]*(?:license|MIT|GPL|Apache|BSD)[^)]*\)", "", desc, flags=re.IGNORECASE)
    
    # 4. Handle sentence casing
    if desc:
        # Convert to sentence case (first letter upper, rest lower)
        desc = desc[0].upper() + desc[1:].lower()
        
        # Preserve standard terms & proper nouns using boundary-safe regexes
        # Sort terms by length descending to replace longer terms first
        for term in sorted(PRESERVE_CASING_TERMS, key=len, reverse=True):
            pattern = rf"(?<![a-zA-Z]){re.escape(term)}(?![a-zA-Z])"
            desc = re.sub(pattern, term, desc, flags=re.IGNORECASE)
        
    # 5. Append commercial label if applicable
    is_commercial = is_commercial_override
    if not is_commercial:
        desc_lower = desc.lower()
        for kw in COMMERCIAL_KEYWORDS:
            if re.search(kw, desc_lower):
                is_commercial = True
                break
                
    if is_commercial and not desc.endswith("(commercial)"):
        # Strip trailing dot if present before adding
        if desc.endswith("."):
            desc = desc[:-1].strip()
        desc = f"{desc} (commercial)"
        
    # 6. Ensure ending dot is placed properly (unless ending in "(commercial)" where dot should be inside or outside based on preference, style guide says "(commercial)" at the end)
    if not desc.endswith(".") and not desc.endswith("(commercial)"):
        desc = desc + "."
    elif desc.endswith("(commercial)") and not desc.endswith(". (commercial)") and not desc.endswith("(commercial)."):
        # Make sure the period sits before the commercial note or at the end
        # Style guide: "Directly states the primary function in sentence case. (commercial)"
        desc = re.sub(r"\s*\(commercial\)", ". (commercial)", desc)
        
    return desc

def format_row(name, url, description, tags, is_commercial=False):
    cleaned_desc = clean_description(description, is_commercial)
    # Tags format: joined by comma-space
    tag_str = ", ".join(tags) if isinstance(tags, list) else tags
    return f"| [{name}]({url}) | {cleaned_desc} | {tag_str} |"

if __name__ == "__main__":
    # Test items
    tests = [
        {
            "name": "MLAtom",
            "url": "https://github.com/dralgroup/mlatom",
            "desc": "A python package for running ML-based atomistic simulations under MIT license.",
            "tags": ["Code/Sim", "Code/ML"],
            "commercial": False
        },
        {
            "name": "Rescale",
            "url": "https://rescale.com",
            "desc": "Official implementation of proprietary HPC cloud orchestration software.",
            "tags": ["Code/WF"],
            "commercial": True
        },
        {
            "name": "FermiLink",
            "url": "https://github.com/TaoELi/FermiLink",
            "desc": "dft-based interface to evaluate electronic couplings in SE(3) crystalline lattices.",
            "tags": ["Code/Lib"],
            "commercial": False
        }
    ]

    print("| Item (URL) | Description | Tags |")
    print("| :--- | :--- | :--- |")
    for t in tests:
        row = format_row(t["name"], t["url"], t["desc"], t["tags"], t["commercial"])
        print(row)
