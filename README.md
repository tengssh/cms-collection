# Computational Materials Science (CMS) Collection

This repository is a curated collection of resources for computational materials science (CMS).

Given the rapidly evolving nature of this field, the proposed categories are designed to be as simple as possible, while being as comprehensive as necessary.

## Table of Contents

- [Curated Lists](#Curated-Lists)
- [Databases & Datasets](#Databases--Datasets)
- [Computing & Workflows](#Computing--Workflows)
- [Machine Learning](#Machine-Learning)
- [Tools: Crystal structures](#Tools-Crystal-structures)
- [Tools: Molecular structures](#Tools-Molecular-structures)
- [Toolkits](#Toolkits)
- [OCW](#OCW)
- [Glossary](#Glossary)
- [License](#License)

## Curated Lists

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [awesome-materials-informatics](https://github.com/tilde-lab/awesome-materials-informatics) | A curated list of known efforts in materials informatics. | List |
| [awesome-matchem-datasets](https://github.com/blaiszik/awesome-matchem-datasets) | A curated list of datasets in materials science for AI/ML. | List, Data |
| [data-resources-for-materials-science](https://github.com/sedaoturak/data-resources-for-materials-science) | A curated list of databases, datasets, and books/handbooks of materials properties for ML applications. | List, Data | 
| [atomistic.software](https://atomistic.software/) | atomistic.software tracks the citation trends of all major atomistic simulation engines. | List, Code/Sim, App |
| [Wannier Software Ecosystem Registry](https://wannier-developers.github.io/wannier-ecosystem-registry/) | Registry of software codes in the Wannier software ecosystem. | List, Code/Sim, App |
| [Electronic Structure Library](https://esl.cecam.org/en/index.html) | A collection of community-maintained libraries and packages for electronic structure simulations. | List, Code/Sim |
| [Existing Workflow systems](https://s.apache.org/existing-workflow-systems) | A curated list of computational workflow systems, engines, and tools for bioinformatics, data analysis, HPC, and scientific computing. | List, Code/WF |
| [WorkflowHub](https://workflowhub.eu/) | Registry for sharing and publishing scientific computational workflows. | List, App |
| [Workflows Community Systems](https://workflows.community/systems) | Community directory and registry of computational workflow systems and execution engines. | List, App |

## Databases & Datasets

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Materials structures & properties](./docs/data-exp-comp.md) | Experimental, computational, and hybrid databases for crystal structures, chemical spectra, and physical properties. | List, Data/Exp, Data/Comp, App |
| [Potentials, basis sets & parameters](./docs/data-potentials-parameters.md) | Pseudopotentials, basis sets, exchange-correlation functionals, and interatomic potentials for atomistic simulations. | List, Data/Comp, Code/Lib, App |
| [Data infrastructure & ontologies](./docs/data-infrastructure.md) | Multidisciplinary data registries, repositories, domain ontologies, and metadata standards. | List, Data, App |

## Computing & Workflows

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Multiscale & Multiphysics](./docs/sim-multiscale-multiphysics.md) | Software packages, tools, and platforms for multiscale and multiphysics materials modeling and simulation. | List, Code/Sim, Code/WF, Code/Lib, Code/ML, App |
| [Integrated workflows](./docs/wf-toolkits.md) | Integrated workflows and management tools for materials science research. | List, Code/WF, Code/Sim, Code/ML, App |

## Machine Learning

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Predictive models](./docs/ml-predictive-models.md) | Machine learning interatomic potentials (MLIPs) and predictive models for material property prediction. | List, Code/ML, Code/Lib |
| [Generative models](./docs/ml-generative-models.md) | Machine learning models for generating crystal structures, molecular structures, spectra, etc. | List, Code/ML, Code/Lib, Code/WF |
| [Uncertainty quantification](./docs/ml-uncertainty-quantification.md) | Uncertainty quantification, active learning, and optimization tools for materials science. | List, Code/ML, Code/Lib, Code/WF |
| [Benchmarks](./docs/ml-benchmarks.md) | Benchmarks for machine learning models and applications in materials science. | List, Data, Code/ML, App |
| [Machine learning toolkits](./docs/ml-toolkits.md) | General-purpose machine learning toolkits for materials science. | List, Code/ML, Code/Lib, Code/WF, App |

## Tools: Crystal structures

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Open Visualization Tool (OVITO)](https://www.ovito.org/) | A visualization tool for particle-based simulations. | Code/Lib, App |
| [Visualization for Electronic and STructural Analysis (VESTA)](https://www.jp-minerals.org/vesta/en/) | A visualization tool for electron densities and crystal morphologies. | App |
| [Atomsk](https://atomsk.univ-lille.fr/) | A command-line program tool to generate structure files for atomic-scale simulations. | Code/Lib |
| [cif2cell](https://github.com/torbjornbjorkman/cif2cell) | A Python package to create structures for electronic structure calculations. | Code/Lib |
| [PyXtal](https://github.com/MaterSim/PyXtal) | A Python package for atomic and molecular crystals. | Code/Lib |
| [Ab initio random structure searching (AIRSS)](https://airss-docs.github.io/) | A tool for generating structures for random structure searching in ab initio calculations. | Code/Lib |
| [Ab-initio Interface Materials Simulation Project for Grain Boundaries (AIMSGB)](https://github.com/ksyang2013/aimsgb) | A Python package for generating periodic grain boundary structures. | Code/Lib |
| [SPuDS - Structure Prediction Diagnostic Software](https://lufaso.domains.unf.edu/spuds/index.html) | A software tool for generating crystal structures of perovskites, including tilting the octahedra. | Code/Lib |
| [SimplySQS](https://github.com/bracerino/atat-sqs-gui) | An interactive Python package for generating special quasi-random structures (SQS). | Code/Lib, App |
| [xrayutilities](https://github.com/dkriegner/xrayutilities) | A collection of scripts for analyzing and simulating X-ray diffraction data. | Code/Lib |
| [GenL](https://github.com/scatterer/GenL) | A fitting tool for X-ray diffraction data on single crystal films. | Code/Lib, App |
| [MOFBuilder](https://github.com/chenxili01/MOFBuilder) | A Python package for building Metal-Organic Framework (MOF) structures. | Code/Lib |
| [pyscal](https://github.com/pyscal/pyscal) | Python library for calculation of local atomic structural environment. | Code/Lib |
| [CALYPSO](https://www.calypso.cn/home/) | Crystal structure prediction using particle swarm optimization. | Code/Sim, App |
| [virp](https://github.com/andypaulchen/virp) | Virtual cell generation from crystal structures containing site disorder. | Code/Lib |

## Tools: Molecular structures

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [IQmol](http://iqmol.org/) | A visualization tool for molecular systems. | App |
| [PyMOL](https://github.com/schrodinger/pymol-open-source) | A visualization tool for molecular systems. | App, Code/Lib |
| [TRajectory Analyzer and VISualizer (TRAVIS)](http://www.travis-analyzer.de/) | A visualization tool for molecular trajectories. | Code/Lib |
| [Visual Molecular Dynamics (VMD)](https://www.ks.uiuc.edu/Research/vmd/) | A visualization tool for molecular systems. | App, Code/Lib |
| [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) | A visualization and analysis program for molecular systems. | App, Code/Lib |
| [PES-trotter](https://github.com/srampinogroup/PES-trotter) | A cross-platform, open-source application built on the Godot Engine for the 3D visualization and exploration of Potential Energy Surfaces (PES). | Code/Lib, App |
| [PACKMOL](https://m3g.github.io/packmol/) | A software tool for packing molecules in defined regions of space, considering short-range repulsions. | Code/Lib |
| [Martini_mapping](https://github.com/eliobaby/Martini_mapping), [Martini_mapper](https://github.com/eliobaby/Martini_mapper) | A Python package for generating coarse-grained models from SMILES strings. | Code/Lib |

## Toolkits

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [Bilbao Crystallographic Server ](https://www.cryst.ehu.es/) | An online server providing programs and utilities for crystallography and solid state materials. | Data, Code/Lib, App |
| [ISOTROPY Software Suite](https://iso.byu.edu/isotropy.php) | A collection of software using group theory to analyze phase transitions in crystalline solids. | Code/Lib, App |
| [Atomic Simulation Environment](https://ase-lib.org/) | A Python toolkit for atomistic simulations. | Code/Lib, Code/ML |
| [Pymatgen (Python Materials Genomics)](https://pymatgen.org/index.html) | A Python library for analyzing materials. | Code/Lib, Code/ML |
| [rDock](https://github.com/CBDD/rDock) | A program for docking small molecules to proteins and nucleic acids. | Code/Sim, Code/Lib |
| [mendeleev](https://github.com/lmmentel/mendeleev/) | A Python package for accessing properties of elements and isotopes from the periodic table of elements. | Code/Lib |
| [Open Babel](https://github.com/openbabel/openbabel) | A toolbox for handling different formats of chemical data. | Code/Lib |
| [VASP Transition State Theory (TST) Tools](https://theory.cm.utexas.edu/vtsttools/) | A collection of scripts and code extensions for applying transition state theory (Nudged Elastic Band, Dimer, etc.) in VASP. | Code/Lib, Code/Sim |
| [bader](https://github.com/henkelmangroup/bader) | Bader charge and electronic population analysis from electronic charge density grids. | Code/Lib |
| [Phonopy](https://github.com/phonopy/phonopy/) | A Python package for phonon calculations of harmonic and quasi-harmonic properties. | Code/Lib |
| [Phono3py](https://github.com/phonopy/phono3py) | A Python package for phonon-phonon interactions related properties. | Code/Lib |
| [Cheminfo](https://www.cheminfo.org/) | A platform with a collection of web applications for visualizing, analyzing, and organizing cheminformatics data. | App, Data |
| [QMatSuite](https://github.com/QMatSuite/QMatSuite) | Graphical user interface for the Quantum ESPRESSO ab-initio simulation suite. | App, Code/Lib |
| [LOBSTER](https://schmeling.ac.rwth-aachen.de/cohp/index.php) | Chemical-bonding analysis including Crystal Orbital Hamilton Population (COHP) and Overlap Population (COOP) from plane-wave DFT outputs. | Code/Lib |
| [LobsterPy](https://github.com/JaGeo/LobsterPy) | Automatic bonding analysis and feature generation from [Lobster](https://schmeling.ac.rwth-aachen.de/cohp/index.php) calculations. | Code/Lib, Code/ML |
| [arpespythontools](https://github.com/pranabdas/arpespythontools) | Explore, analyze, and visualize Angle-Resolved Photoemission Spectroscopy (ARPES) data. | Code/Lib |
| [MatCalc](https://github.com/materialyzeai/matcalc) | Calculating materials properties from potential energy surfaces using machine learning interatomic potentials and DFT. | Code/Lib, Code/ML |
| [auto-kappa](https://github.com/phonix-db/auto-kappa) | Automated calculation of anharmonic phonon properties. | Code/WF, Code/Lib |
| [AMDAT](https://github.com/dssimmons-codes/AMDAT) | C++ toolkit for post-processing molecular dynamics trajectories, with a focus on static and dynamic analyses of amorphous, glassy, and polymer materials. | Code/Lib |
| [PLUMED2](https://github.com/plumed/plumed2) | Free energy calculations, enhanced-sampling algorithms, and trajectory analysis for molecular dynamics simulations. | Code/Lib, Code/Sim |
| [DL_FIELD](https://www.ccp5.ac.uk/dl_field/) | Force field construction and conversion utility for molecular simulations (DL_POLY, GROMACS, LAMMPS). | Code/Lib |
| [LAMMPS-AST](https://github.com/ethanholbrook/LAMMPS-AST) | Sanitizing, parsing, and transforming LAMMPS input scripts into abstract syntax trees (ASTs) for linting, validation, and workflow integration. | Code/Lib |

## OCW

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [The Space Group List Project](https://crystalsymmetry.wordpress.com/2014/08/15/the-space-group-list-project-as-a-poster/) | A gallery for the collection of 3D crystal examples for all 230 space groups. | Edu |
| [MolSSI Education](https://education.molssi.org/) | Tutorials on programming, software development, and molecular modeling by the Molecular Sciences Software Institute (MolSSI). | Edu |
| [Psi4Education](https://psicode.org/posts/psi4education/) | A collection of Jupyter Notebook labs for quantum chemistry by Psi4. | Edu, App |
| [Ising simulation](https://mattbierbaum.github.io/ising.js/) | Interactive browser-based simulation of the 2D Ising model. | Edu, App |
| [Soft Matter Demos](https://softmatterdemos.org/) | Interactive simulation demos for soft matter physics. | Edu, App |
| [Landau theory](https://jeffjar.me/statmech/fun.html) | Lecture notes and interative demos for Landau theory. | Edu, App |
| [Phonon website](https://henriquemiranda.github.io/phononwebsite/index.html) | An interactive website to visualize lattice vibrations and phonon dispersions (with extensions to wave functions and charge density) of materials. | App, Edu |
| [ML-in-chemistry-101](https://github.com/BingqingCheng/ML-in-chemistry-101) | A graduate-level course for machine learning in chemistry. | Edu, Code/ML |
| [matgenb](https://github.com/materialyzeai/matgenb) | A collection of Jupyter notebooks for materials science. |  Edu, Code/Sim, Code/Lib |
| [AI4Chemistry](https://schwallergroup.github.io/ai4chem_course/) | A hands-on course covering cheminformatics and machine learning for chemistry. | Edu, Code/ML |
| [Computational Materials Physics](https://www.compmatphys.org/) | A free online course on Density Functional Theory, with hands-on exercises based on Quantum ESPRESSO software. | Edu, App |
| [Modeling Materials Using Density Functional Theory](https://github.com/jkitchin/dft-book) | A repository of learning resources for Density Functional Theory (DFT) using VASP and ASE. | Edu, App |
| [nanoHUB](https://nanohub.org/) | An online platform providing various browser-based simulation tools and educational resources in nanotechnology and materials science. | Edu, App |
| [Lhumos](https://www.lhumos.org/) | An online learning platform for modelling and simulation of matters in computational materials science. | Edu |
| [OSSCAR Course Applications](https://www.osscar.org/courses/index.html) | Educational resources on quantum mechanics and materials science, developed by the OSSCAR (Open Software Services for Classrooms and Research) Team. | Edu, App |
| [The Atomistic Cookbook](https://atomistic-cookbook.org/index.html) | Computational recipes for modeling matter at the atomic scale, featuring interactive guides and templates for simulations. | Edu, App |
| [IBM Quantum Learning](https://quantum.cloud.ibm.com/learning/en) | Learning resources on quantum computing by IBM. | Edu |
| [Google Quantum AI](https://quantumai.google/resources) | Learning resources on quantum computing by Google. | Edu |
| [Microsoft Quantum](https://quantum.microsoft.com/en-us/insights/education) | Learning resources on quantum computing by Microsoft. | Edu |
| [The Carpentries Lessons](https://carpentries.org/lessons/) | Foundational coding and data science tutorials for researchers. | Edu |
| [CodeRefinery lessons](https://coderefinery.org/lessons/) | Lessons on essential software development practices for computational scientists. | Edu |
| [BestPractices](https://github.com/anthony-wang/BestPractices) | Best practices for materials informatics research. | Edu |

## Glossary

| Tag       | Description          |
| --------- | -------------------- |
| List      | Curated compilations |
| Data      | Data & metadata      |
| Data/Exp  | Experimental data    |
| Data/Comp | Computational data   |
| Code/Lib  | Pre-/Post-processing |
| Code/Sim  | Simulation engines   |
| Code/WF   | Workflow tools       |
| Code/ML   | AI/ML derived        |
| App       | Apps & web services  |
| Edu       | Educational/learning |

> Go to [Table of Contents](#Table-of-Contents)

## License

<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/cc-zero.png" width="100">

> While the curated collection and overall contents of this page are dedicated to the public domain (CC0), the copyright and specific usage instructions for individual items remain with their respective creators. Users must refer to the original source page for licensing terms and requirements for attribution or reuse.
