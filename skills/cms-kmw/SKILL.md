---
name: Computational Materials Science Knowledge Management Workflow (CMS-KMW)
description: A workflow for curating, validating, classifying, and cataloging scientific software, databases, machine learning models, and other related resources in computational materials science.
---

# Computational Materials Science Knowledge Management Workflow

This skill defines a guidelines-based workflow intended to assist in curating, validating, and cataloging computational materials science (CMS) resources. It aims to evaluate if indexed resources are scientifically relevant, active, secure, and consistently categorized.

## 1. Overview
Knowledge Management Workflow (KMW) in CMS involves indexing and organizing software packages, workflow managers, datasets, machine learning models and potentials (MLIPs), simulation platforms, utility tools, databases, curated lists, and educational resources. A key objective is to maintain a high-quality, comprehensive repository of resources for the scientific community, aiming to support diverse computational materials informatics, modeling, and simulation tasks.

## 2. Core Curation Workflow

```mermaid
graph TD
    A[New Resource Submission] --> B[Step 1: Scrape Metadata & Readme]
    B --> C[Step 2: Content Validation & Critique]
    C --> D[Step 3: Security & Maintenance Screening]
    D --> E{Meets Scope & Criteria?}
    E -- No --> F[Relegate to Backlog]
    E -- Yes --> G[Step 4: Classification & Tagging]
    G --> H[Step 5: Cataloging & Description Styling]
    H --> I[Insert into Target Markdown File]
```

### Step 1: Metadata & Readme Retrieval (Use [fetch_metadata.py](./scripts/fetch_metadata.py))
- Scrape or query repository API (e.g., GitHub, Gitee) to retrieve description, license, active branches, and commit history.
- Retrieve the repository's `README.md` to extract technical functionality and developer intent.

### Step 2: Content Validation & Scientific Critique (Use [check_scientific_rigor.py](./scripts/check_scientific_rigor.py))
- **Functional Accuracy:** Verify that the catalog description matches actual code functionality.
- **Scientific Rigor:** Evaluate the tool's peer review standing (e.g., associated journal papers, citations).
- **Reproducibility:** Check if standard methods or protocols are clearly defined.
- **Community Standing:** Assess usage within the niche community (e.g., DFT, molecular dynamics, machine learning informatics).

### Step 3: Security & Maintenance Screening (Use [check_security_maintenance.py](./scripts/check_security_maintenance.py))
- **Domain Verification:** Verify that links point to recognized domains (such as GitHub, GitLab, Gitee, Zenodo, or established university/consortium sites).
- **Maintenance Activity:** Check for recent commits or active issue tracking to avoid "abandonware."
- **Licensing:** Prioritize resources with standard open-source licenses (MIT, Apache 2.0, BSD, GPL). Explicitly label commercial platforms.

### Step 4: Classification & Structural Placement (Use [classify_resource.py](./scripts/classify_resource.py))
Determine the file target based on architectural purpose and domain role:
1. **Databases & Datasets (`data-*.md`):**
   - **Structures & Properties (`docs/data-exp-comp.md`):** Databases of experimental, computational, or hybrid crystal structures, spectra, and thermophysical properties (`Data/Exp`, `Data/Comp`).
   - **Potentials & Parameters (`docs/data-potentials-parameters.md`):** Pseudopotential libraries, basis sets, exchange-correlation functionals, and interatomic force field repositories (`Data/Comp`, `Code/Lib`).
   - **Data Infrastructure (`docs/data-infrastructure.md`):** Multidisciplinary data registries, domain ontologies, metadata schemas, and database interoperability APIs (`Data`, `App`).
2. **Computing & Workflows (`sim-*.md`, `wf-*.md`):**
   - **Multiscale & Multiphysics (`docs/sim-multiscale-multiphysics.md`):** Simulation engines, ab initio DFT/MD solvers, and multiscale/concurrent scale coupling packages (`Code/Sim`).
   - **Integrated Workflows (`docs/wf-toolkits.md`):** Workflow managers, pipeline orchestrators, agentic automation systems, and MCP servers (`Code/WF`).
3. **Machine Learning (`ml-*.md`):**
   - **Predictive Models (`docs/ml-predictive-models.md`):** Machine learning interatomic potentials (MLIPs) and property prediction models (`Code/ML`).
   - **Generative Models (`docs/ml-generative-models.md`):** Generative crystal/molecular structure models and inverse materials design (`Code/ML`).
   - **Uncertainty Quantification (`docs/ml-uncertainty-quantification.md`):** UQ, active learning loops, and Bayesian optimization tools (`Code/ML`).
   - **Benchmarks (`docs/ml-benchmarks.md`):** Evaluation harnesses, benchmark datasets, and model leaderboards (`Data`, `Code/ML`).
   - **Machine Learning Toolkits (`docs/ml-toolkits.md`):** General ML development frameworks, molecular descriptors, and graph neural network libraries (`Code/ML`, `Code/Lib`).
4. **Tools & Toolkits (`tools-*.md`):**
   - **Crystal Structures (`docs/tools-crystal-structures.md`):** Pre/post-processing tools for crystal structure generation, space group analysis, and 3D visualizers (`Code/Lib`, `App`).
   - **Molecular Structures (`docs/tools-molecular-structures.md`):** Molecular modeling, packing, trajectory visualization, and coarse-graining tools (`Code/Lib`, `App`).
   - **Simulation & Analysis (`docs/tools-simulation-analysis.md`):** Core computational libraries (ASE, Pymatgen), phonon solvers, electronic structure analyzers, and sampling plugins (`Code/Lib`, `Code/Sim`).
5. **Educational Resources (`ocw-*.md`):**
   - **Educational Resources (`docs/ocw-cms.md`):** Academic courses, interactive simulation applets, computational cookbooks, and research computing lessons (`Edu`).
6. **Out-of-Scope Backlog (`backlog.md`):**
   - Resources that are general-purpose (e.g., non-materials specific tools), archived/abandoned repositories, or raw dataset publications (`docs/backlog.md`).

### Step 5: Cataloging & Description Styling (Use [format_catalog_entry.py](./scripts/format_catalog_entry.py))
Adhere to the following stylistic guidelines when formatting entries:
*   **Objectivity:** Avoid qualitative marketing adjectives (e.g., "state-of-the-art", "highly efficient", "advanced").
*   **Redundancy Removal:** Avoid introductory packaging prefixes (e.g., "A Python package for...", "Code for...", "Official implementation of..."). Start directly with the core action or function.
*   **No License Info in Text:** Exclude license types from the description string, as users have to check the license accordingly.
*   **Sentence Case:** Capitalize only the first character of the description sentence (excluding proper nouns, acronyms, or math terms like `SE(3)`, `DFT`, `XANES`).
*   **Commercial Labeling:** Append `(commercial)` at the end of the description if the platform serves commercial/proprietary purposes.

---

## 3. Reference Table Format
Entries must be formatted as rows in Markdown tables:
```markdown
| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Name](URL) | Directly states the primary function in sentence case. (commercial) | Tag1, Tag2 |
```

---

## 4. Standard Tag Taxonomy
Assign tags strictly from the repository glossary:
*   `List`: Curated compilations
*   `Data`: General data & metadata registries
*   `Data/Exp`: Experimental datasets
*   `Data/Comp`: Theoretically calculated structures/properties
*   `Code/Lib`: Pre-processing, post-processing, and utility libraries
*   `Code/Sim`: Solvers and simulation engines
*   `Code/WF`: Workflow managers and pipeline orchestrators
*   `Code/ML`: Machine learning architectures, models, and MLIPs
*   `App`: Web services and graphical application portals
*   `Edu`: Tutorials and educational open courseware

---

## 5. Helper Scripts
This skill is supported by the following automation scripts located in the [scripts/](./scripts) directory:
- [fetch_metadata.py](./scripts/fetch_metadata.py): Automates Step 1 by fetching repository description, license, last update date, and readme file contents using GitHub/Gitee APIs.
- [check_scientific_rigor.py](./scripts/check_scientific_rigor.py): Automates Step 2 by analyzing readme texts for citations, DOIs, arXiv numbers, BibTeX entries, reproducibility setups, and scientific keywords.
- [check_security_maintenance.py](./scripts/check_security_maintenance.py): Automates Step 3 by checking domain trust lists, performing live URL checks, verifying open-source licenses, checking archived states, and calculating update intervals.
- [classify_resource.py](./scripts/classify_resource.py): Automates Step 4 by matching keyword heuristics to determine target categories, target markdown files, and standard tags.
- [format_catalog_entry.py](./scripts/format_catalog_entry.py): Automates Step 5 by removing description redundancies, enforcing sentence case (while preserving scientific terms), detecting commercial features, and outputting formatted markdown table rows.

---

## 6. Example Execution Prompts
Here are examples of how a user triggers this workflow:

- *"Add the repository https://github.com/dralgroup/mlatom to the collection, verifying its scope and classification."*
- *"We need to add a new conversational agent platform Rescale. Can you evaluate it under the CMS knowledge management workflow?"*
- *"Format the description of ASAP to match the repository styling rules."*

