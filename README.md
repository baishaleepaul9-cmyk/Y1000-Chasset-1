# Y1000-Chasset-1
# Phylogeny-Aware Genome–Phenotype Integration for Yeast Chassis Prioritization

## Overview

This repository contains the computational analyses supporting the study:

**"[Insert manuscript title]"**

The study integrates comparative genomic information, phenotypic measurements, machine-learning modelling, functional interpretation, network analysis, and multi-trait Pareto optimization to prioritize candidate yeast chassis from the Y1000+ collection.

The study analysed **420 yeast taxa** using three quantitative phenotypes:

- Carbon-source breadth
- Nitrogen-source breadth
- Utilized median growth

The final genome–phenotype modelling framework used **79 robust BUSCO features**.

---

## Repository Scope

This repository contains the **final analytical components and principal outputs** of the study, specifically:

1. Final machine-learning model integration
2. Functional interpretation
3. Pathway analysis
4. BUSCO–pathway network analysis
5. Pareto-based candidate prioritization
6. Principal result tables and/or figures supporting these analyses

The repository does **not** contain the complete end-to-end preprocessing pipeline.

---

## Initial Data Processing and BUSCO Analysis

The initial genomic data processing, genome preparation, and BUSCO-based feature generation were performed in a **Linux environment using command-line bioinformatics tools and workflows**.

The BUSCO analyses and associated preprocessing steps generated the genomic feature matrix subsequently used for machine-learning analysis. These initial Linux-based command-line workflows are **not included as scripts in this repository**.

The resulting processed genomic feature matrix and the subsequently selected robust BUSCO feature set were used for the downstream analyses provided here.

This distinction is important because the present repository is intended to document and reproduce the **final integration and downstream analytical stages**, rather than to provide a complete reconstruction of every upstream bioinformatics command used during genome processing.

---

## Final ML Model Integration

The final genome–phenotype modelling used the 79 robust BUSCO features across 420 yeast taxa.

| Phenotype | Feature set | Final model | OOF R² | Mean phylogenetic-fold R² | Permutation P |
|---|---|---|---:|---:|---:|
| Carbon-source breadth | Robust_79 | Random Forest | 0.0036 | −0.2771 | 0.0198 |
| Nitrogen-source breadth | Robust_79 | Random Forest | −0.0949 | −0.1871 | 0.9703 |
| Utilized median growth | Robust_79 | Elastic Net | −0.0319 | −0.1520 | 0.9802 |

Phylogeny-aware cross-validation and permutation testing were used to assess the robustness of the genome–phenotype relationships.

Carbon-source breadth showed the clearest evidence of a non-random genomic signal, although its low predictive magnitude indicates that the analysed genomic features explain only a limited component of phenotypic variation.

---

## Functional and Network Analysis

The downstream functional analyses provided biological context for the genomic features associated with the genome–phenotype analyses.

These analyses included:

- BUSCO functional annotation
- Functional category analysis
- KEGG pathway associations
- Biological theme identification
- BUSCO–pathway network construction
- Network-level interpretation

The network analysis was used for functional interpretation and should not be interpreted as establishing direct causal relationships between individual BUSCO features and phenotypes.

---

## Pareto-Based Candidate Prioritization

The phenotypic dimensions were subsequently integrated using multi-trait Pareto optimization.

The analysis considered:

- Carbon-source breadth
- Nitrogen-source breadth
- Utilized median growth

This analysis identified **11 Pareto-optimal candidate taxa** from the 420-taxon dataset.

The Pareto candidates represent non-dominated combinations of the three phenotypic traits rather than a single universally optimal phenotype.

The resulting taxa therefore represent **computationally prioritized candidates for experimental investigation**, rather than experimentally validated microbial chassis.

---

## Dataset

The genomic and phenotypic information used in the study was obtained from the publicly available **Y1000+ yeast collection**.

Y1000+ Project:

https://myco-lb.jgi.doe.gov/yeasts-1000/

Y1000+ genome resources:

https://doi.org/10.25452/figshare.plus.c.6714042

The present study used 420 taxa for which the required genomic and phenotypic information was available.

The original Y1000+ genomic resources are not redistributed in this repository.

---

## Reproducibility and Pipeline Coverage

The repository should be interpreted according to the following workflow:

**Y1000+ genomic resources**  
↓  
**Linux-based genome processing and BUSCO analysis**  
*(upstream scripts not included)*  
↓  
**420-taxon genomic feature matrix**  
↓  
**79 robust BUSCO features**  
↓  
**Final ML model integration**  
↓  
**Functional/pathway analysis**  
↓  
**BUSCO–pathway network analysis**  
↓  
**Multi-trait Pareto optimization**  
↓  
**11 candidate yeast taxa**

The scripts provided in this repository correspond primarily to the **final ML integration and downstream biological and candidate-prioritization analyses**.

---

## Important Note on Missing Upstream Scripts

The absence of the BUSCO and initial genome-processing scripts does not indicate that these analyses were not performed. These steps were completed separately in a Linux command-line environment as part of the upstream genomic data-processing workflow.

The repository therefore focuses on the analyses that generated the final integrated results presented in the manuscript.

---

## Interpretation of Results

The final analysis demonstrated trait-dependent genome–phenotype correspondence.

Carbon-source breadth produced the clearest non-random signal under permutation testing:

- OOF R² = 0.0036
- Mean phylogenetic-fold R² = −0.2771
- Permutation P = 0.0198

Nitrogen-source breadth and utilized median growth showed weaker predictive correspondence:

- Nitrogen-source breadth: OOF R² = −0.0949; permutation P = 0.9703
- Utilized median growth: OOF R² = −0.0319; permutation P = 0.9802

These results should therefore not be interpreted as demonstrating strong predictive performance. Instead, they indicate that the detectable genomic contribution differs among phenotypic traits.

The Pareto analysis provided a complementary approach for prioritizing taxa based on their joint phenotypic characteristics.

---



## License

[Insert appropriate license here.]
