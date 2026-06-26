"""
Step 2: Content Validation & Scientific Critique
Analyzes a repository's README and description to verify scientific rigor, 
reproducibility indicators, academic citations, and domain keywords.
"""

import re

# Key CMS/Chemistry scientific terms to verify domain alignment
SCIENTIFIC_KEYWORDS = [
    r"\bdft\b", r"density\s+functional\s+theory", r"molecular\s+dynamics", 
    r"ab\s+initio", r"tight\s+binding", r"dftb", r"electronic\s+structure", 
    r"interatomic\s+potential", r"machine\s+learning\s+potential", r"\bmlip\b",
    r"force\s+field", r"crystallography", r"\bcif\b", r"\bxrd\b", r"\bxanes\b",
    r"qm/mm", r"quantum\s+chemistry", r"coarse-grained", r"monte\s+carlo",
    r"thermodynamics", r"band\s+gap", r"density\s+of\s+states"
]

def check_scientific_rigor(name, description, readme_content):
    report = {
        "resource": name,
        "has_citations": False,
        "citation_references": [],
        "domain_keywords_found": [],
        "has_reproducibility_indicators": False,
        "reproducibility_details": [],
        "scientific_score": 0.0,
        "verdict": "Low Rigor"
    }

    # 1. Search for Citations/Publications (DOIs, arXiv, bibtex, paper keywords)
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b"
    arxiv_pattern = r"\barXiv:\d{4}\.\d{4,5}\b"
    
    dois = re.findall(doi_pattern, readme_content)
    arxivs = re.findall(arxiv_pattern, readme_content)
    
    if dois:
        report["has_citations"] = True
        report["citation_references"].extend([f"DOI: {d}" for d in set(dois)])
    if arxivs:
        report["has_citations"] = True
        report["citation_references"].extend([f"arXiv: {a}" for a in set(arxivs)])
        
    citation_keywords = ["citation", "cite this", "reference", "bibtex", "paper", "journal", "publication"]
    found_cit_kw = [kw for kw in citation_keywords if kw in readme_content.lower()]
    if found_cit_kw:
        report["has_citations"] = True
        report["citation_references"].append(f"Contains citation keywords: {', '.join(found_cit_kw)}")

    # 2. Check for domain-specific keywords
    combined_text = f"{description} {readme_content}".lower()
    for kw in SCIENTIFIC_KEYWORDS:
        if re.search(kw, combined_text):
            report["domain_keywords_found"].append(kw.replace(r"\b", "").replace(r"\s+", " "))
            
    # 3. Check for Reproducibility Indicators (install steps, run examples, test suites)
    reprod_patterns = {
        "installation": r"(pip install|conda install|mamba install|setup\.py|poetry add|cmake)",
        "run_example": r"(example|usage|tutorial|how to run|getting started|run\.sh|python -m)",
        "testing": r"(pytest|unittest|tox|test suite|run tests)"
    }
    
    for category, pattern in reprod_patterns.items():
        if re.search(pattern, combined_text):
            report["has_reproducibility_indicators"] = True
            report["reproducibility_details"].append(category)

    # 4. Rigor Scoring & Verdict Heuristics
    score = 0.0
    if report["has_citations"]:
        score += 4.0
    if len(report["domain_keywords_found"]) >= 2:
        score += 3.0
    elif len(report["domain_keywords_found"]) == 1:
        score += 1.5
    if report["has_reproducibility_indicators"]:
        score += 3.0
        
    report["scientific_score"] = score
    if score >= 7.0:
        report["verdict"] = "High Rigor (Strong scientific base & documentation)"
    elif score >= 4.0:
        report["verdict"] = "Moderate Rigor (Acceptable, needs manual sanity check)"
    else:
        report["verdict"] = "Low Rigor (Unverified/unreproducible scientific contribution)"

    return report

if __name__ == "__main__":
    # Example usage: mlatom dummy readme
    example_readme = """
    # MLAtom: Platform for AI-enhanced Chemistry
    
    MLAtom provides interface for various interatomic potentials and density functional theory methods.
    To cite MLAtom, please use:
    Pevzner et al., Journal of Chemical Theory and Computation 2020 16 (12), 7955-7970.
    DOI: 10.1021/acs.jctc.0c00888
    
    ## Installation
    pip install mlatom
    
    ## Usage
    mlatom run.py --method=ANI-1x
    """
    
    res = check_scientific_rigor(
        "mlatom", 
        "Package for AI-enhanced computational chemistry and machine learning potentials.",
        example_readme
    )
    
    print(f"Evaluation for {res['resource']}:")
    print(f"Verdict: {res['verdict']}")
    print(f"Scientific Score: {res['scientific_score']}/10.0")
    print(f"Citations Found: {res['citation_references']}")
    print(f"Domain Keywords: {res['domain_keywords_found']}")
    print(f"Reproducibility Checks: {res['reproducibility_details']}")
