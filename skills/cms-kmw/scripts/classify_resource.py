"""
Step 4: Classification & Tagging
Heuristic categorizer that maps computational materials science tools to their 
target markdown files and taxonomic tags based on textual analysis.
"""

import re

# Keywords representing simulation, workflow, ML toolkit, and backlog boundaries
SIMULATION_KEYWORDS = [
    r"solver", r"dft", r"molecular\s+dynamics", r"tight\s+binding", r"dftb", 
    r"quantum\s+chemistry", r"ab\s+initio", r"simulation\s+engine", r"hartree",
    r"schrodinger", r"potentials", r"forcefield", r"mc", r"monte\s+carlo",
    r"multiscale", r"scale\s+coupling", r"qmm", r"qm/mm", r"coarse\s+grained"
]

WORKFLOW_KEYWORDS = [
    r"workflow", r"orchestrat", r"pipeline", r"agent", r"llm", r"chat", 
    r"automation", r"assistant", r"database\s+manager", r"remote\s+launch",
    r"framework\s+for\s+running", r"process\s+manager", r"run\s+manager",
    r"active\s+learning\s+loop", r"gui", r"web\s+app", r"saas", r"portal"
]

ML_TOOLKIT_KEYWORDS = [
    r"machine\s+learning\s+toolkit", r"ml\s+toolkit", r"training\s+pipeline", 
    r"representation", r"descriptor", r"neural\s+network\s+potential",
    r"graph\s+neural", r"gnn", r"fitting", r"developer\s+library", 
    r"tensor", r"torch", r"jax", r"scikit"
]

OUT_OF_SCOPE_KEYWORDS = [
    r"pde\s+solver", r"finite\s+element", r"navier\s+stokes", r"fluid\s+dynamics",
    r"image\s+generation", r"bioinformatics", r"genomics", r"astrophysics",
    r"robotics", r"general\s+physics", r"publication", r"dataset\s+record"
]

def classify_and_tag(name, description, readme_content):
    combined_text = f"{name} {description} {readme_content}".lower()
    desc_lower = description.lower()
    
    # 1. Determine targets
    sim_hits = sum(1 for kw in SIMULATION_KEYWORDS if re.search(kw, combined_text))
    wf_hits = sum(1 for kw in WORKFLOW_KEYWORDS if re.search(kw, combined_text))
    ml_hits = sum(1 for kw in ML_TOOLKIT_KEYWORDS if re.search(kw, combined_text))
    out_hits = sum(1 for kw in OUT_OF_SCOPE_KEYWORDS if re.search(kw, combined_text))
    
    # 2. File Destination Suggestion & Primary Tag
    target_file = ""
    primary_tag = ""
    rationale = ""
    
    if out_hits > 1 or (out_hits >= 1 and sim_hits == 0 and wf_hits == 0):
        target_file = "misc/backlog.md"
        primary_tag = "Backlog"
        rationale = "Contains general/out-of-scope engineering or publication keywords."
    elif wf_hits > sim_hits and wf_hits > ml_hits:
        target_file = "multiscale/workflows.md"
        primary_tag = "Code/WF"
        rationale = "Focuses on pipeline, automation, agent orchestration, or execution managers."
    elif ml_hits > sim_hits:
        target_file = "machine_learning/ml_toolkits.md"
        primary_tag = "Code/ML"
        rationale = "Focuses on ML frameworks, representations, or model development toolkits."
    else:
        # Defaults to Simulation/Solvers
        target_file = "multiscale/multiscale.md"
        primary_tag = "Code/Sim"
        rationale = "Bundles atomistic solvers, calculations (DFT/MD), or concurrent scale solvers."

    # 3. Collect all matching tags
    tags = []
    
    # Add List tag if description indicates a curated list
    if "curated list" in desc_lower or "awesome list" in desc_lower:
        tags.append("List")
        
    # Check for Edu tag
    if "tutorial" in desc_lower or "course" in desc_lower or "lecture" in desc_lower:
        tags.append("Edu")
        
    # Check for Data tags
    # Only assign Data as primary/secondary if it's primarily a dataset or contains dataset keywords
    is_dataset = "dataset" in desc_lower or "database" in desc_lower or "repository of" in desc_lower
    if is_dataset or (("dataset" in combined_text or "database" in combined_text) and primary_tag == "Backlog"):
        if "experimental" in combined_text:
            tags.append("Data/Exp")
        elif "calculated" in combined_text or "dft" in combined_text or "quantum" in combined_text:
            tags.append("Data/Comp")
        else:
            tags.append("Data")
            
    # Add primary software tag
    if primary_tag != "Backlog":
        tags.append(primary_tag)
        
        # Add secondary code tags if there are significant hits
        if primary_tag != "Code/Sim" and sim_hits > 1:
            tags.append("Code/Sim")
        if primary_tag != "Code/WF" and wf_hits > 1:
            tags.append("Code/WF")
        if primary_tag != "Code/ML" and ml_hits > 1:
            tags.append("Code/ML")
            
        # Add Code/Lib as default library helper unless it's only a solver/engine
        if "library" in combined_text or "package" in combined_text or "toolkit" in combined_text or "wrapper" in combined_text:
            tags.append("Code/Lib")
            
    # Check for App tag (web service, SaaS, GUI, portal)
    if "web service" in combined_text or "gui portal" in combined_text or "web app" in combined_text or "saas" in combined_text or "web interface" in combined_text:
        tags.append("App")
        
    # De-duplicate while preserving order
    unique_tags = []
    for t in tags:
        if t not in unique_tags:
            unique_tags.append(t)
            
    # Fallback to List/Backlog if no tags assigned
    if not unique_tags:
        unique_tags = [primary_tag]
        
    return {
        "resource": name,
        "recommended_file": target_file,
        "recommended_tags": unique_tags,
        "rationale": rationale,
        "hits": {
            "simulation": sim_hits,
            "workflow": wf_hits,
            "ml_toolkit": ml_hits,
            "out_of_scope": out_hits
        }
    }

if __name__ == "__main__":
    test_inputs = [
        {
            "name": "MLAtom",
            "desc": "AI-enhanced computational chemistry program for atomistic simulations.",
            "readme": "MLAtom bundles multiple semiempirical and DFT solvers, offering full atomistic potentials."
        },
        {
            "name": "Masgent",
            "desc": "A multi-agent workflow platform for materials science.",
            "readme": "Masgent uses LLM agents to automate computational material science pipelines and execute DFT codes."
        },
        {
            "name": "MIST",
            "desc": "Machine learning interatomic potential fitting package.",
            "readme": "Python framework for fitting machine learning interatomic potentials and training descriptors."
        },
        {
            "name": "SciExplorer",
            "desc": "General scientific PDE solver and image processing utility.",
            "readme": "Solving Navier-Stokes and general partial differential equations using neural network approximations."
        }
    ]

    for item in test_inputs:
        print("=" * 60)
        res = classify_and_tag(item["name"], item["desc"], item["readme"])
        print(f"Name: {res['resource']}")
        print(f"Recommended Target File: {res['recommended_file']}")
        print(f"Recommended Tags: {', '.join(res['recommended_tags'])}")
        print(f"Rationale: {res['rationale']}")
        print(f"Keywords hits: Sim={res['hits']['simulation']}, WF={res['hits']['workflow']}, ML={res['hits']['ml_toolkit']}, Out={res['hits']['out_of_scope']}")
