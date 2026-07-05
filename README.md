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
| [Electronic Structure Library](https://esl.cecam.org/en/index.html) | A collection of community-maintained libraries and packages for electronic structure simulations. | List, Code/Sim |
| [Existing Workflow systems](https://s.apache.org/existing-workflow-systems) | A curated list of computational workflow systems, engines, and tools for bioinformatics, data analysis, HPC, and scientific computing. | List, Code/WF |
| [Workflows Community Systems](https://workflows.community/systems) | Community directory and registry of computational workflow systems and execution engines. | List, App |

## Databases & Datasets

| Item (URL) | Description | Tags |
| :--------- | :---------- | :--- |
| [International Tables for Crystallography](https://it.iucr.org/) | The definitive resource and reference work for crystallography, providing standardized data on crystal symmetry, space groups, and diffraction methods. | Data |
| [re3data\.org](https://www.re3data.org/) | A registry of multidisciplinary research data repositories. | Data, App |
| [Crystallography Open Database (COD)](https://www.crystallography.net/cod/) | A collection of crystal structures of organic, inorganic, metal-organic compounds and minerals, excluding biopolymers. (open-access) | Data/Exp, App |
| [Theoretical Crystallography Open Database (TCOD)](https://www.crystallography.net/tcod/) | A collection of theoretically calculated or refined crystal structures of organic, inorganic, metal-organic compounds and minerals, excluding biopolymers. (open-access) | Data/Comp, App |
| [Anyterial](https://www.anyterial.se/) | A portal for materials databases powered by the [High-Throughput Toolkit (httk)](https://httk.org/). | Data/Comp |
| [Open Materials Database](https://openmaterialsdb.se/) | A database of material properties for computational materials design research. | Data/Comp, App |
| [Automatic Defect Analysis and Qualification (ADAQ)](https://defects.anyterial.se/) | A platform for automatic workflows for high-throughput calculations of point defects in semiconductors. | Data/Comp, App |
| [NOMAD Encyclopedia](https://nomad-lab.eu/prod/rae/encyclopedia/) | A periodic table search interface for materials data, including electronic structure calculations and material properties. | Data/Comp, App |
| [Access Structures](https://www.ccdc.cam.ac.uk/structures/) | A service provided by the CCDC and FIZ Karlsruhe to view and retrieve structures, including CSD and ICSD datasets. (free) | Data/Exp, App |
| [Inorganic Crystal Structure Database (ICSD)](https://icsd.fiz-karlsruhe.de/index.xhtml) | A comprehensive repository for fully determined inorganic and intermetallic crystal structures. | Data/Exp, App |
| [Pearson's Crystal Data (PCD)](https://www.crystalimpact.com/pcd/) | A curated repository originating from the Pauling File project, covering hundreds of thousands of inorganic compounds. (commercial) | Data, App |
| [Organic Materials Database (OMDB)](https://omdb.mathub.io/) | A comprehensive computational database for the electronic properties of 3D organic crystals, provided by the Nordita theoretical quantum matter group. (open-access) | Data/Comp, App |
| [Inorganic Material Database (AtomWork)](https://crystdb.nims.go.jp) | A database for crystal structure, x-ray diffraction, material properties and phase diagram data of inorganic and metallic materials. | Data, App |
| [NIMS Materials Database (MatNavi)](https://mits.nims.go.jp/) | A massive, multi-domain materials data platform, integrating experimental and computational datasets for polymers, metals, and inorganic compounds with built-in property prediction tools. | Data, App |
| [Materials Data Repository (MDR)](https://mdr.nims.go.jp/) | A data repository for materials informatics that integrates research papers and presentations with materials data. | Data, App |
| [Polymer Database (PoLyInfo)](https://polymer.nims.go.jp/) | A database for polymers that includes information on processing methods, measurement conditions, and material properties (e.g., chemical, thermal, electrical, and mechanical properties). | Data, App |
| [Database of Zeolite Structures](https://www.iza-structure.org/databases/) | A database of validated zeolite structures featuring frameworks, NMR spectra, and channel system analysis. | Data, App |
| [Materials Project](https://materialsproject.org/) | A comprehensive database for inorganic materials featuring interactive analysis web interfaces. | Data/Comp, App |
| [Open Quantum Materials Database (OQMD)](https://oqmd.org/) | A massive materials database of millions of DFT calculations, including thermodynamic and structural properties. | Data/Comp, App |
| [OQMD+](https://hse.oqmd.org/) | A subset of the OQMD featuring hybrid functional (HSE) calculations, providing more accurate band gaps and electronic properties for inorganic materials. | Data/Comp, App |
| [Computational Materials Repository (CMR)](https://cmr.fysik.dtu.dk/) | A collection of project-specific materials databases with DFT datasets in standardized ASE-database formats. | Data |
| [Open Catalyst Project](https://opencatalystproject.org/) | A massive open-source repository of DFT trajectories and benchmark tasks for catalysts. | Data, App |
| [Quantum MOF (QMOF) Database](https://github.com/arosen93/QMOF) | A specialized dataset of quantum-chemical properties for metal–organic frameworks (MOFs) and coordination polymers derived from high-throughput periodic DFT calculations. | Data/Comp |
| [MofasaDB](https://mofux.ai/explore) | An annotated dataset with generated MOF (Metal-Organic Framework) structures, trained on [QMOF](https://github.com/arosen93/QMOF). | Data/Comp, App |
| [Materials Platform for Data Science](https://mpds.io/) | A curated large-scale database of experimental inorganic materials data, extracted from scientific publications. | Data, App |
| [matterverse.ai](https://matterverse.ai/) | A large-scale repository of millions of predicted, stable inorganic crystals generated by machine learning models ([M3GNet](https://github.com/materialsvirtuallab/m3gnet)). | Data/Comp, App |
| [Topological Materials Database](https://topologicalquantumchemistry.org/) | A specialized database for topological materials, characterized based on the theory of topological quantum chemistry. | Data/Comp, App |
| [Open Databases Integration for Materials Design (OPTIMADE)](https://www.optimade.org/) | A consortium-driven REST API standard for the interoperability of materials databases. | App, Data |
| [Hydrogen Materials—Advanced Research Consortium (HyMARC)](https://datahub.hymarc.org/) | A data hub for solid-state hydrogen storage research, including experimental and computational datasets. | Data, App |
| [MSI Eureka](https://search.msi-eureka.com/search) | A large collection of inorganic phase diagrams and constitutional data of binary and multi-component systems. | Data, App |
| [Open Reaction Database](https://open-reaction-database.org/client/search) | A database for structured organic reaction data to support data-driven discovery and automated synthesis. (open-access) | Data, App |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | A large open chemistry database, containing millions of structures linked to biological activity, toxicity, physical properties, and patents. | Data, App |
| [NIST Materials Data Repository](https://materialsdata.nist.gov/) | A repository for the open exchange and preservation of diverse materials datasets and models. (open-access) | Data, App |
| [NLRMatDB](https://materials.nrel.gov/) | A computational database for renewable energy materials, including photovoltaics, thermoelectrics, and water-splitting catalysts. | Data, App |
| [Active Thermochemical Tables (ATcT)](https://atct.anl.gov/) | A specialized database using the Thermochemical Network (TN) approach to combine experimental and state-of-the art theoretical data into highly reliable thermodynamic values. | Data, App |
| [Spectral Database for Organic Compounds (SDBS)](https://sdbs.db.aist.go.jp/) | A comprehensive database of experimental spectra for organic compounds, featuring electron impact Mass spectrum (EI-MS), Fourier transform infrared spectrum (FT-IR), 1H nuclear magnetic resonance (NMR), 13C NMR, Raman, and electron spin resonance (ESR) spectrum. | Data/Exp, App |
| [Network Database System for Thermophysical Property Data](https://tpds.db.aist.go.jp/) | A comprehensive repository of experimental thermophysical properties for solids, liquids, and melts, including thermal conductivity, diffusivity, and surface tension. | Data/Exp, App |
| [Solid-State NMR Spectral Database (SSNMR_SD)](https://ssnmr-sd.db.aist.go.jp/SSNMR/Top.php) | A specialized experimental database of solid-state NMR spectra mainly for solid samples. | Data/Exp, App |
| [Alexandria](https://alexandria.icams.rub.de/) | A massive database of consistent DFT-computed properties for inorganic solids. (open-access) | Data/Comp |
| [Materials Cloud - Discover](https://www.materialscloud.org/discover/) | A curated collection of interactive research datasets with tailored visualizations, maintained by the Materials Cloud team. | App, Data |
| [CHEMnetBASE](https://www.chemnetbase.com/) | A collection of chemical databases covering physical constants, organic compound properties, and thermodynamic data. (proprietary) | Data, App |
| [3D (VRML) The Fermi Surface Database](https://www.phys.ufl.edu/fermisurface/) | An interactive gallery of 3D Fermi surfaces for ~45 elemental solids (Al, Fe, Cu, etc.). | Data, App, Edu |
| [2D Materials Encyclopedia](http://www.2dmatpedia.org/) | An open database for thousands of 2D materials from both top-down (exfoliation from bulk Materials Project) and bottom-up (elemental substitution) approaches. | Data, App |
| [Starrydata2](https://www.starrydata2.org/) | An open materials database extracted materials data from plot images. | Data, App |
| [facebook/OMAT24](https://huggingface.co/datasets/facebook/OMAT24) | A massive dataset of  millions of density functional theory calculations for inorganic materials, featuring both structural relaxations and non-equilibrium trajectories. | Data/Comp |
| [facebook/OMol25+OPoly26](https://huggingface.co/facebook/OMol25) | A comprehensive, open dataset of millions of high-accuracy density functional theory calculations, including electronic densities, wavefunctions, and molecular orbital information. The Open Polymers 2026 (OPoly26) Dataset is an extension of the Open Molecules 2025 (OMol25) Dataset. | Data/Comp |
| [SandboxAQ/aqcat25-dataset](https://huggingface.co/datasets/SandboxAQ/aqcat25-dataset) | A large-scale, spin-aware dataset with diverse DFT calculation trajectories for catalyst systems. | Data/Comp |
| [DeFecTdb](https://db-amdis.org/defectdb) | A collection of DFT datasets of radiation-induced defect structures in materials for nuclear fusion/fission applications. | Data/Comp, App |
| [ASM Databases](https://www.asminternational.org/materials-resources/online-databases/) | A databases of peer-reviewed engineering materials databases, featuring alloys, properties, phase diagrams, microstructures, and failure analysis (subscription access). | Data |
| [MAGNDATA](https://cryst.ehu.es/magndata/) | A database of more than 2000 published commensurate and incommensurate magnetic structures with portable cif-type files. | Data, App |
| [DataScribe.Cloud](https://datascribe.cloud/) | An AI-powered platform hosted by the Materials Science Department of Texas A&M University, accessible via [DataScribe Python API](https://github.com/DataScribe-Cloud/datascribe_api). | Data, App |
| [Thermodynamic DataBase DataBase (TDB DB)](https://avdwgroup.engin.brown.edu/) | A specialized search engine indexing freely available thermodynamic data in TDB format from scientific literature (CALPHAD journal supplements, NIMS, NIST) for CALPHAD modeling and high-throughput materials discovery workflows. | Data, App |
| [RCSB Protein Data Bank (RCSB PDB)](https://www.rcsb.org/) | A repository for experimentally determined 3D structure data for large biological molecules (proteins, DNA, and RNA). | Data/Exp, App |
| [Worldwide PDB (wwPDB)](https://www.wwpdb.org/) | The Worldwide Protein Data Bank (wwPDB) archive is a freely accessible repository of 3D structures of proteins, nucleic acids and complex assemblies. | Data, App |
| [Thermodynamic Databases](https://thermocalc.com/products/databases/) | A collection of CALPHAD-optimized databases for predicting phase equilibria, thermodynamic properties, and behaviors in alloys and multicomponent systems. (commercial & academic) | Data, Code/Sim |
| [Knovel](https://app.knovel.com/kn) | A platform for engineering references with interactive tools for data analysis and material searches. (commercial) | Data, App |
| [MatWeb](https://www.matweb.com/index.aspx) | A database of materials information for engineering materials, including metals, plastics, ceramics, and composites, with tools for common engineering tasks. (free & premium services) | Data, App |
| [SpringerMaterials](https://materials.springer.com/) | A curated database of materials and physical/chemical properties with interactive data visualization and analysis. (commercial) | Data, App |
| [ChemSpider](https://www.chemspider.com/) | A database for millions of chemical structures, properties, identifiers, and links, supporting SMILES/InChI text string searches. (free) | Data, App |
| [Libxc](https://libxc.gitlab.io/) | A library of exchange-correlation functionals for density-functional theory. | Data/Comp, Code/Lib |
| [Standard solid-state pseudopotentials (SSSP)](https://www.materialscloud.org/discover/sssp/) | A collection of solid-state pseudopotentials optimized for precision or efficiency. | Data/Comp, Code/Lib, App |
| [PseudoDojo.org](https://www.pseudo-dojo.org/) | A repository of pseudopotentials for various density-functional theory codes. | Data/Comp, Code/Lib, App |
| [Basis Set Exchange (BSE)](https://www.basissetexchange.org/) | A repository of basis sets for computational chemistry calculations. | Data/Comp, Code/Lib, App |
| [Pseudopotential Library](https://pseudopotentiallibrary.org/) | A repository of pseudopotentials for quantum Monte Carlo and quantum chemistry. | Data/Comp, Code/Lib, App |
| [Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/) | A repository of interatomic potentials (force fields) for various materials. | Data/Comp, Code/Lib, App |
| [Open Knowledgebase of Interatomic Models (OpenKIM)](https://openkim.org/) | A curated repository of interatomic potentials for atomistic simulations. | Data/Comp, Code/Lib, Code/ML, App |
| [Elementary Multiperspective Material Ontology (EMMO)](https://emmo-repo.github.io/) | A standardized representational ontology framework for materials modelling and characterization knowledge. | Data |
| [Material Core (MatCore)](https://matcore.org/) | A community-specific metadata standard for computational materials science. | Data |

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
| [Phonopy](https://github.com/phonopy/phonopy/) | A Python package for phonon calculations of harmonic and quasi-harmonic properties. | Code/Lib |
| [Phono3py](https://github.com/phonopy/phono3py) | A Python package for phonon-phonon interactions related properties. | Code/Lib |
| [Cheminfo](https://www.cheminfo.org/) | A platform with a collection of web applications for visualizing, analyzing, and organizing cheminformatics data. | App, Data |
| [QMatSuite](https://github.com/QMatSuite/QMatSuite) | Graphical user interface for the Quantum ESPRESSO ab-initio simulation suite. | App, Code/Lib |
| [LOBSTER](https://schmeling.ac.rwth-aachen.de/cohp/index.php) | Chemical-bonding analysis including Crystal Orbital Hamilton Population (COHP) and Overlap Population (COOP) from plane-wave DFT outputs. | Code/Lib |
| [LobsterPy](https://github.com/JaGeo/LobsterPy) | Automatic bonding analysis and feature generation from [Lobster](https://schmeling.ac.rwth-aachen.de/cohp/index.php) calculations. | Code/Lib, Code/ML |


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
