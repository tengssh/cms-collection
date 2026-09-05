"""
Step 4: Classification & Tagging
Heuristic categorizer that maps computational materials science tools to their 
target markdown files and taxonomic tags based on textual analysis.
"""

import re

# Comprehensive keyword regexes covering the entire repository taxonomy

EDU_KEYWORDS = [
    r"course", r"lecture", r"tutorial", r"syllabus", r"open\s+courseware",
    r"learning\s+platform", r"educational", r"lesson", r"cookbook",
    r"interactive\s+demo", r"statmech\s+demo", r"classroom"
]

DATA_POTENTIALS_KEYWORDS = [
    r"pseudopotential", r"basis\s+set", r"exchange-correlation", r"functional\s+library",
    r"interatomic\s+potential\s+repository", r"force\s+field\s+repository", r"openkim",
    r"paw\s+dataset", r"\bpotcar\b", r"\bupf\b"
]

DATA_INFRASTRUCTURE_KEYWORDS = [
    r"ontology", r"metadata\s+standard", r"\bprov-o\b", r"\bemmo\b", r"\bcmso\b", r"\basmo\b",
    r"\boptimade\b", r"\bmatportal\b", r"data\s+registry", r"fair\s+data",
    r"multidisciplinary\s+repository\s+registry", r"semantic\s+web"
]

DATA_MATERIALS_KEYWORDS = [
    r"materials\s+database", r"materials\s+project", r"\boqmd\b", r"\bicsd\b", r"\bcod\b",
    r"crystal\s+structure\s+database", r"dft\s+calculations\s+database",
    r"spectroscopic\s+database", r"thermophysical\s+database", r"phase\s+diagram\s+database",
    r"reaction\s+database", r"catalyst\s+dataset", r"calphad\s+database"
]

CRYSTAL_TOOL_KEYWORDS = [
    r"crystal\s+structure", r"space\s+group", r"\bcif\b", r"\bcif2", r"supercell",
    r"grain\s+boundary", r"perovskite", r"random\s+structure\s+search",
    r"crystal\s+visualization", r"xrd\s+fitting", r"x-ray\s+diffraction", r"\bvesta\b", r"\bovito\b",
    r"\batomsk\b", r"site\s+disorder"
]

MOLECULAR_TOOL_KEYWORDS = [
    r"molecular\s+visualization", r"trajectory\s+visualizer", r"molecule\s+packing",
    r"packing\s+molecules", r"\bpackmol\b", r"molecular\s+system", r"small\s+molecule",
    r"coarse\s+grained", r"\bmartini\b", r"\bvmd\b", r"\bpymol\b", r"\bchimerax\b", r"\bsmiles\b",
    r"docking", r"conformation"
]

ML_BENCHMARK_KEYWORDS = [
    r"benchmark", r"leaderboard", r"evaluation\s+suite", r"test\s+harness",
    r"evaluation\s+of", r"benchmark\s+dataset"
]

ML_UQ_KEYWORDS = [
    r"uncertainty\s+quantification", r"active\s+learning", r"bayesian\s+optimization",
    r"conformal\s+prediction", r"ensemble\s+uncertainty"
]

ML_GENERATIVE_KEYWORDS = [
    r"generative\s+model", r"crystal\s+generation", r"crystal\s+diffusion",
    r"molecular\s+generation", r"inverse\s+design", r"denoising\s+diffusion",
    r"autoregressive\s+generation"
]

ML_PREDICTIVE_KEYWORDS = [
    r"machine\s+learning\s+interatomic\s+potential", r"mlip", r"property\s+prediction",
    r"universal\s+potential", r"neural\s+network\s+potential", r"force\s+prediction",
    r"deepmd", r"mace", r"chgnet", r"allegro"
]

ML_TOOLKIT_KEYWORDS = [
    r"machine\s+learning\s+toolkit", r"ml\s+toolkit", r"training\s+pipeline", 
    r"representation", r"descriptor", r"graph\s+neural", r"gnn",
    r"torch\s+extension", r"jax\s+library"
]

WORKFLOW_KEYWORDS = [
    r"workflow", r"orchestrat", r"pipeline", r"agent", r"llm\s+agent", r"mcp\s+server",
    r"automation", r"execution\s+manager", r"run\s+manager", r"aiida", r"jobflow"
]

SIMULATION_ANALYSIS_TOOLKIT_KEYWORDS = [
    r"atomistic\s+simulation\s+environment", r"pymatgen", r"phonon", r"phonopy",
    r"bader\s+charge", r"crystal\s+orbital", r"cohp", r"lobster", r"post-processing",
    r"free\s+energy", r"enhanced\s+sampling", r"plumed", r"electronic\s+population"
]

SIMULATION_KEYWORDS = [
    r"dft\s+code", r"molecular\s+dynamics\s+engine", r"tight\s+binding\s+solver",
    r"quantum\s+chemistry\s+package", r"multiscale", r"scale\s+coupling",
    r"qm/mm", r"pde\s+finite\s+element", r"carrier\s+transport", r"boltzmann\s+transport",
    r"ab\s+initio\s+simulation", r"electronic\s+structure\s+code"
]


def classify_and_tag(name, description, readme_content=""):
    combined_text = f"{name} {description} {readme_content}".lower()
    desc_lower = description.lower()
    
    # 1. Count heuristic keyword matches
    scores = {
        "edu": sum(1 for kw in EDU_KEYWORDS if re.search(kw, combined_text)),
        "data_potentials": sum(1 for kw in DATA_POTENTIALS_KEYWORDS if re.search(kw, combined_text)),
        "data_infrastructure": sum(1 for kw in DATA_INFRASTRUCTURE_KEYWORDS if re.search(kw, combined_text)),
        "data_materials": sum(1 for kw in DATA_MATERIALS_KEYWORDS if re.search(kw, combined_text)),
        "crystal_tool": sum(1 for kw in CRYSTAL_TOOL_KEYWORDS if re.search(kw, combined_text)),
        "molecular_tool": sum(1 for kw in MOLECULAR_TOOL_KEYWORDS if re.search(kw, combined_text)),
        "ml_benchmark": sum(1 for kw in ML_BENCHMARK_KEYWORDS if re.search(kw, combined_text)),
        "ml_uq": sum(1 for kw in ML_UQ_KEYWORDS if re.search(kw, combined_text)),
        "ml_generative": sum(1 for kw in ML_GENERATIVE_KEYWORDS if re.search(kw, combined_text)),
        "ml_predictive": sum(1 for kw in ML_PREDICTIVE_KEYWORDS if re.search(kw, combined_text)),
        "ml_toolkit": sum(1 for kw in ML_TOOLKIT_KEYWORDS if re.search(kw, combined_text)),
        "workflow": sum(1 for kw in WORKFLOW_KEYWORDS if re.search(kw, combined_text)),
        "sim_analysis": sum(1 for kw in SIMULATION_ANALYSIS_TOOLKIT_KEYWORDS if re.search(kw, combined_text)),
        "simulation": sum(1 for kw in SIMULATION_KEYWORDS if re.search(kw, combined_text)),
    }
    
    # 2. Determine target file and primary classification
    target_file = ""
    rationale = ""
    tags = []
    
    # High-priority categories based on specific domain markers
    if scores["edu"] >= 1 and ("course" in desc_lower or "tutorial" in desc_lower or "demo" in desc_lower or "cookbook" in desc_lower):
        target_file = "docs/ocw-cms.md"
        tags.extend(["Edu"])
        rationale = "Educational resource, academic course, interactive applet, or tutorial workshop."
    elif scores["data_potentials"] >= 1:
        target_file = "docs/data-potentials-parameters.md"
        tags.extend(["Data/Comp", "Code/Lib"])
        rationale = "Pseudopotential library, basis set exchange, functional library, or interatomic potential repository."
    elif scores["data_infrastructure"] >= 1:
        target_file = "docs/data-infrastructure.md"
        tags.extend(["Data"])
        rationale = "Data repository registry, domain ontology, metadata schema, or data federation API."
    elif scores["data_materials"] >= 1 or "database" in desc_lower or "dataset" in desc_lower:
        target_file = "docs/data-exp-comp.md"
        if "experimental" in combined_text or "spectra" in combined_text:
            tags.append("Data/Exp")
        elif "dft" in combined_text or "calculated" in combined_text or "computational" in combined_text:
            tags.append("Data/Comp")
        else:
            tags.append("Data")
        rationale = "Database or dataset of materials structures, properties, or calculations."
    elif scores["ml_benchmark"] >= 1:
        target_file = "docs/ml-benchmarks.md"
        tags.extend(["Data", "Code/ML"])
        rationale = "Benchmarking suite, evaluation framework, or model leaderboard."
    elif scores["ml_generative"] >= 1:
        target_file = "docs/ml-generative-models.md"
        tags.extend(["Code/ML", "Code/Lib"])
        rationale = "Generative model for crystal/molecular structure creation or inverse materials design."
    elif scores["ml_uq"] >= 1:
        target_file = "docs/ml-uncertainty-quantification.md"
        tags.extend(["Code/ML", "Code/Lib"])
        rationale = "Uncertainty quantification, active learning, or Bayesian optimization framework."
    elif scores["ml_predictive"] >= 1:
        target_file = "docs/ml-predictive-models.md"
        tags.extend(["Code/ML", "Code/Lib"])
        rationale = "Machine learning interatomic potential (MLIP) or material property predictor."
    elif scores["crystal_tool"] >= 1 and scores["simulation"] == 0:
        target_file = "docs/tools-crystal-structures.md"
        tags.extend(["Code/Lib"])
        rationale = "Pre/post-processing or visualization tool for crystal structures and periodic systems."
    elif scores["molecular_tool"] >= 1 and scores["simulation"] == 0:
        target_file = "docs/tools-molecular-structures.md"
        tags.extend(["Code/Lib"])
        rationale = "Modeling, packing, or trajectory visualization tool for molecular systems."
    elif scores["workflow"] > scores["simulation"] and scores["workflow"] > scores["ml_toolkit"]:
        target_file = "docs/wf-toolkits.md"
        tags.extend(["Code/WF"])
        rationale = "Workflow manager, agentic orchestration framework, or pipeline tool."
    elif scores["sim_analysis"] >= 1:
        target_file = "docs/tools-simulation-analysis.md"
        tags.extend(["Code/Lib"])
        rationale = "Atomistic simulation toolkit, phonon solver, bonding analyzer, or post-processing library."
    elif scores["ml_toolkit"] > scores["simulation"]:
        target_file = "docs/ml-toolkits.md"
        tags.extend(["Code/ML", "Code/Lib"])
        rationale = "General machine learning toolkit, representation builder, or descriptor library."
    elif scores["simulation"] >= 1:
        target_file = "docs/sim-multiscale-multiphysics.md"
        tags.extend(["Code/Sim"])
        rationale = "Simulation engine, multiscale solver, or scale-coupling framework."
    else:
        # Relegate to backlog if no computational materials science criteria matched
        target_file = "docs/backlog.md"
        tags.append("Backlog")
        rationale = "No domain criteria matched within the computational materials science taxonomy; relegated to backlog for curator review."

    # 3. Supplemental tag deductions
    if target_file != "docs/backlog.md":
        if "curated list" in desc_lower or "awesome list" in desc_lower:
            tags.insert(0, "List")
        if "app" in combined_text or "web service" in combined_text or "web app" in combined_text or "gui" in combined_text or "portal" in combined_text:
            tags.append("App")
        if "library" in combined_text or "package" in combined_text or "toolkit" in combined_text:
            if "Code/Lib" not in tags and "Code/Sim" in tags:
                tags.append("Code/Lib")

    # Deduplicate while preserving order
    unique_tags = []
    for t in tags:
        if t not in unique_tags:
            unique_tags.append(t)
            
    return {
        "resource": name,
        "recommended_file": target_file,
        "recommended_tags": unique_tags,
        "rationale": rationale,
        "hits": scores
    }

if __name__ == "__main__":
    test_cases = [
        {
            "name": "VESTA",
            "desc": "A visualization tool for electron densities and crystal morphologies.",
            "readme": "3D visualization program for structural models and volumetric data like electron/nuclear densities."
        },
        {
            "name": "PACKMOL",
            "desc": "Initial configuration generator for molecular dynamics simulations by packing molecules.",
            "readme": "Packmol creates initial configurations for molecular dynamics in defined spatial regions."
        },
        {
            "name": "Materials Project",
            "desc": "Open database for inorganic materials property calculations.",
            "readme": "High-throughput DFT calculation database providing thermodynamic and electronic properties."
        },
        {
            "name": "Basis Set Exchange",
            "desc": "Repository of basis sets for quantum chemistry calculations.",
            "readme": "Standard library and web portal for Gaussian basis sets."
        },
        {
            "name": "EMMO",
            "desc": "Standardized representational ontology framework for materials modeling.",
            "readme": "Elementary Multiperspective Material Ontology for materials science knowledge representation."
        },
        {
            "name": "MACE",
            "desc": "Higher order equivariant message passing neural network potentials.",
            "readme": "Fast, accurate machine learning interatomic potentials trained for materials and molecules."
        },
        {
            "name": "Computational Materials Physics",
            "desc": "A free online course on Density Functional Theory with Quantum ESPRESSO.",
            "readme": "Course syllabus, lecture notes, and hands-on exercises for DFT."
        },
        {
            "name": "AiiDA",
            "desc": "Automated interactive infrastructure and daemon for computational materials science.",
            "readme": "Robust workflow management and data provenance orchestration framework."
        },
        {
            "name": "Django",
            "desc": "A high-level Python web framework that encourages rapid development.",
            "readme": "Django is a web framework that makes it easier to build web apps quickly with less code."
        }
    ]

    for item in test_cases:
        print("=" * 70)
        res = classify_and_tag(item["name"], item["desc"], item["readme"])
        print(f"Resource: {res['resource']}")
        print(f"Target File: {res['recommended_file']}")
        print(f"Tags: {', '.join(res['recommended_tags'])}")
        print(f"Rationale: {res['rationale']}")
