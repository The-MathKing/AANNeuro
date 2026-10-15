# Multi-Cohort Transcriptomic Benchmark and Confound Audit in Alzheimer's Disease

**Author:** Aryan Padarthi  
**Affiliation:** Allen High School, Allen, Texas  
**Contact:** aryan.padarthi@student.allenisd.org  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NCBI GEO](https://img.shields.io/badge/NCBI_GEO-GSE84422%20|%20GSE118553%20|%20GSE132903-green.svg)](https://www.ncbi.nlm.nih.gov/geo/)

---

## Overview

This repository hosts the code, statistical pipelines, and experimental results for a multi-cohort, leakage-aware analysis of post-mortem human brain transcriptomics in Alzheimer's disease (AD). The framework evaluates **1,649 tissue samples across 425 donors** from three distinct biobank cohorts:

1. **MSBB (GSE84422):** 19 brain regions across 118 donors (Braak stages 0–VI, CDR 0–5, CERAD).
2. **Discovery Cohort (GSE118553):** 4 brain regions across 112 donors (Control, AsymAD, AD).
3. **Lockbox Cohort (GSE132903):** 195 middle temporal gyrus samples (97 AD vs 98 Controls, matched on age and sex).

Every statistic and metric is computed dynamically from raw data, saved to structured JSON outputs in `results/`, and recorded in an immutable analysis ledger (`ledger.csv`).

---

## Benchmark Hypotheses & Summary of Results

| Hypothesis | Evaluation Protocol | Primary Result | Finding |
| :--- | :--- | :--- | :--- |
| **H1: Regional Vulnerability** | Progression scores tracking anatomical Braak order across 19 regions | Single split: $\rho = 0.406$ ($p = 0.040$)<br/>Repeated CV (10x): $\rho = 0.181$ ($p = 0.230$) | **Unsupported (Null)**: Sampling error in small cohorts ($n \approx 50\text{--}60$); primary motor cortex and frontal pole scored on par with temporal areas. |
| **H2: Cognitive Resilience** | Distinguish high-pathology resilient donors ($\text{Braak} \ge \text{IV}, \text{CDR} \le 0.5$) from demented donors | $N = 7$ resilient donors. Sample transfer AUC = 0.612 ($p = 0.076$). | **Exploratory / Mixed**: Discrete transfer near chance; continuous AsymAD signature tracked residual cognition ($\rho = -0.32, p = 0.0005$). |
| **H3: Lockbox Replication** | Generalize frozen temporal-cortex classifier to untouched external cohort | **Lockbox AUC = 0.855** [0.802, 0.907]<br/>Permutation $p = 0.0002$ | **Confirmed (Replicated)**: Survived age/sex matching, residualization, and neuronal composition controls. |

---

## Confound Audits & Biological Findings

* **E1 (Leakage Benchmark):** Compound leakage inflated AUC by only $\sim 0.015$, indicating that demographic confounding and pseudoreplication present greater real-world threats than leakage alone in high-signal tasks.
* **E2 (Pseudoreplication Audit):** Treating 1,052 multi-region samples as independent donors fabricated an apparent microglial elevation. Aggregated to donor level ($N = 117$), the microglial signal dissolved ($p = 0.134$); only neuronal loss remained significant ($\rho = -0.289, p = 0.0016$).
* **E5 (Synaptic vs. ECM Modules):** Synaptic markers collapsed with Braak progression ($\text{OR} = 21.9, p = 1.3 \times 10^{-4}$), while extracellular matrix and vascular genes were upregulated ($\text{OR} = 6.4, p = 0.039$).
* **E8 (Sparse Panel):** An elastic-net panel identified 20 genes (AUC 0.817 in lockbox), but two genes (*KDM5D*, *TMSB4Y*) were Y-linked sex genes.
* **E9 (Cerebellar Confound):** Cerebellar AD vs. control initially achieved an AUC of 0.961, but AD donors were 11.5 years older ($p = 0.003$). Age residualization reduced the AUC to 0.661.

---

## Repository Structure

```
.
├── config/                 # Pipeline configuration (paths, seeds, thresholds)
├── ledger.csv              # Immutable ledger of all analysis runs & seeds
├── src/
│   ├── load/               # Leakage-free dataset parsers (GSE84422, GSE118553, GSE132903)
│   ├── harmonize/          # Cross-platform probe matching & QC filters
│   ├── eval/               # Donor-grouped CV, cluster bootstrap, permutation engines
│   ├── models/             # Regularized logistic, ridge, PCA, cVAE
│   ├── experiments/        # Deep model and pan-cortical benchmarks
│   └── analyses/           # Execution scripts for H1-H3 and E1-E10 benchmarks
├── results/                # 30+ raw JSON result outputs with parameter hashes
├── figures/                # Publication-quality vector PDF and high-res PNG figures
└── tests/                  # Integrity test suites (leakage, range, checksums)
```

---

## Quickstart & Reproduction

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity

Run the test suite to verify that donor isolation, in-fold transforms, and checksums are strictly respected:

```bash
pytest tests/
```

### 3. Run Analyses and Regenerate Results

```bash
# Execute master analysis battery
python src/analyses/run_all_analyses.py

# Generate all figures
python figures/generate_all_figures.py
```

---

## License

This repository is distributed under the MIT License for open scientific reproduction.
