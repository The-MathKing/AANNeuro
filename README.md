# Replication and Confound Audit of an Alzheimer's Disease Brain Transcriptomic Signature: A Leakage-Aware, Multi-Cohort Analysis

**Author:** Aryan Padarthi  
**Affiliation:** Allen High School, Allen, Texas, USA  
**Contact:** aryan.padarthi@student.allenisd.org  
**Target Submission:** American Academy of Neurology (AAN) Neuroscience Research Prize  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NCBI GEO](https://img.shields.io/badge/NCBI_GEO-GSE84422%20|%20GSE118553%20|%20GSE132903-green.svg)](https://www.ncbi.nlm.nih.gov/geo/)
[![Pre-Registration](https://img.shields.io/badge/Pre--Registration-LOCKED-success.svg)](PREREGISTRATION.md)

---

## Why I Built This Project

I spent about eleven months working on this computational biology project on my own, starting in late 2025. I first got interested in Alzheimer's disease after reading about cognitive resilience in biology class. It was fascinating to learn that some older individuals can live into their nineties with brains full of amyloid plaques and tau tangles, yet stay completely sharp on memory tests right up until death.

When I started reading recent computational neuroscience papers, I kept noticing a pattern: many machine learning models claimed near-perfect diagnostic accuracies (often 95% to 99%) on post-mortem brain tissue. But when those same algorithms were tested on independent patient cohorts from other brain banks, their accuracy almost always collapsed.

Looking closely into why this happens, I found three recurring problems in the published literature:
1. Donor Leakage: Datasets with multiple brain tissue samples from the same person were being split randomly across cross-validation folds. This allowed models to memorize an individual donor's genetic background rather than learning genuine disease biology.
2. Missing External Validation: Classifiers were rarely tested on completely untouched external biobank cohorts.
3. Conflating Pathology with Dementia: Many studies treated post-mortem tissue pathology and clinical dementia as identical, which ignores asymptomatic and resilient individuals entirely.

To see what signals actually hold up under honest conditions, I pre-registered my study protocol (PREREGISTRATION.md) before touching the raw data, gathered 1,649 microarray samples across 19 brain regions from three public biobanks, and wrote the computational pipeline from scratch in Python to enforce strict donor quarantine.

---

## Datasets

All genomic data analyzed in this project comes from public post-mortem human brain cohorts deposited on NCBI Gene Expression Omnibus (GEO):

* Mount Sinai Brain Bank (MSBB / GSE84422):  
  1,053 Affymetrix microarray profiles across 19 brain regions from 118 donors. Each donor includes documented Clinical Dementia Rating scores (CDR 0 to 5), Braak neurofibrillary tangle stages (0 to VI), and CERAD plaque scores.
* Discovery Cohort (GSE118553):  
  401 Illumina microarray profiles across 4 brain regions (temporal cortex, entorhinal cortex, frontal cortex, and cerebellum) from 112 donors, categorized into healthy controls, asymptomatic AD, and symptomatic AD dementia.
* Untouched External Lockbox Cohort (GSE132903):  
  195 middle temporal gyrus samples (97 AD cases and 98 age-matched and sex-matched controls) profiled on the Illumina platform. This cohort remained completely quarantined until the diagnostic model was frozen.

---

## Hypotheses and Findings

Before running any code on the data, I locked down three core hypotheses in my pre-registration protocol:

| Hypothesis | Test Description | What I Found | Scientific Verdict |
| :--- | :--- | :--- | :--- |
| **H1: Regional Vulnerability** | Tested whether gene expression progression scores across 19 brain structures track the anatomical Braak staging order. | An initial single split suggested a correlation of rho = 0.406 (p = 0.040). But when I averaged across 10 repeated donor-grouped cross-validation splits, the correlation dropped to rho = 0.181 (p = 0.230). | **Unsupported (Null Result)**: Regions damaged earliest (such as the hippocampus) hit a biological ceiling of severe neuronal loss and gliosis in autopsy tissue, compressing measurable gene variance. |
| **H2: Cognitive Resilience** | Tested whether transcriptomic signatures separate resilient donors (Braak >= IV, CDR <= 0.5) from demented donors with equal tangle burden. | Only 7 resilient donors existed in the MSBB cohort, making discrete classification underpowered (sample-level AUC 0.612, p = 0.076). However, an asymptomatic AD continuous score correlated with residual cognition after pathology adjustment (rho = -0.34, p = 0.00017). | **Exploratory / Mixed**: Discrete classification requires larger biobank cohorts, but continuous scoring identifies promising individual target genes. |
| **H3: External Lockbox Replication** | Evaluated whether a frozen temporal cortex classifier generalizes to a separate, untouched biobank cohort without retraining. | The frozen model achieved an **AUC of 0.855** (95% CI 0.802 to 0.907; permutation p = 0.0002) in the untouched lockbox cohort. The signal held up after controlling for age, sex, and cellular composition. | **Confirmed (Replicated)**: A robust cortical transcriptomic signature replicates externally when donor leakage and demographic shortcuts are prevented. |

---

## Confound Audits: Where Models Get Fooled

Beyond the three main hypotheses, I ran ten secondary audits (E1 through E10) to figure out how computational biomarker models can be misled:

* Pseudoreplication (E2): Treating 1,052 multi-region tissue samples as independent observations produced a misleadingly strong microglial activation signal (p < 0.001). Once properly clustered by biological donor (N = 117), that microglial signal dissolved (p = 0.134), leaving only neuronal loss significant.
* Demographic Confounding (E9): A cerebellar classifier initially showed an impressive AUC of 0.961. But checking the cohort metadata revealed that the AD donors were on average 11.5 years older than the controls. Once I controlled for age, the AUC dropped to 0.661, proving the classifier was mostly detecting normal biological aging rather than Alzheimer's.
* Automated Feature Selection Traps (E8): Running sparse elastic-net feature selection without biological supervision picked up Y-chromosome genes (like KDM5D and TMSB4Y) simply because of slight sex ratio imbalances between case and control groups.
* Leakage Optimism (E1): Allowing donor overlap between folds inflated cross-validation AUC by about 0.015. However, demographic imbalances and pseudoreplication turned out to be far larger threats to scientific validity.

---

## Repository Structure

```text
.
├── PREREGISTRATION.md      # Study protocol locked prior to data analysis
├── LICENSE                 # MIT License
├── config/                 # Pipeline configuration (seeds, paths, thresholds)
├── ledger.csv              # Audit ledger tracking every computational run
├── src/
│   ├── load/               # Zero-leakage data parsers for GSE84422, GSE118553, GSE132903
│   ├── harmonize/          # Cross-platform probe matching and quality control
│   ├── eval/               # Donor-grouped CV, cluster bootstrap, and permutation engines
│   ├── models/             # Regularized logistic regression, ridge, residualized PCA, cVAE
│   ├── experiments/        # Deep generative models and pan-cortical benchmarks
│   └── analyses/           # Execution scripts for H1-H3 and E1-E10 audits
├── results/                # 30+ structured JSON output files with runtime parameter hashes
├── figures/                # Vector PDF and high-resolution PNG figures
└── tests/                  # Automated pytest suite enforcing zero donor leakage
```

---

## How to Reproduce Everything

### 1. Set Up the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Integrity Tests

Run the automated test suite to confirm that donor isolation, in-fold transformations, and data checksums are strictly respected:

```bash
pytest tests/
```

### 3. Run Analyses and Generate Figures

```bash
# Execute the full analysis pipeline
python src/analyses/run_all_analyses.py

# Generate all figures
python figures/generate_all_figures.py
```

All summary metrics and test statistics flow directly from the machine-generated JSON files in `results/` into LaTeX macros (`paper/numbers.tex`) to eliminate manual transcription errors.

---

## Project Background & Contact

This project was developed by Aryan Padarthi as an independent student research submission to the 2027 American Academy of Neurology (AAN) Neuroscience Research Prize.

* Author: Aryan Padarthi
* School: Allen High School, Allen, Texas, USA
* Email: aryan.padarthi@student.allenisd.org

If you have any questions about the analysis, the pre-registration protocol, or reproducing the results, feel free to open a GitHub issue or send an email.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
