# Study Pre-Registration: Transcriptomic Signatures of Regional Vulnerability and Cognitive Resilience in Alzheimer's Disease

**Date of Pre-Registration:** September 20, 2026  
**Status:** LOCKED / PRE-ANALYSIS  
**Working Title:** *Transcriptomic Signatures of Regional Vulnerability and Cognitive Resilience in Alzheimer's Disease: A Leakage-Aware, Multi-Cohort Analysis*

---

## 1. Study Overview & Primary Hypotheses

This study investigates regional vulnerability, cognitive resilience at matched neuropathological burden, and cross-cohort generalizability in Alzheimer's disease (AD) using strict leakage-aware machine learning and statistical modeling.

### Primary Hypotheses (Confirmatory):

- **Hypothesis 1 (H1 - Regional Vulnerability):**
  Disease-associated transcriptomic progression scores and effect sizes across human brain regions track the regions' a-priori neuropathological (Braak-defined) order of tau neurofibrillary tangle involvement.
  - *Primary Metric:* Spearman rank correlation ($\rho$) between regional effect size/transcriptomic disease score and the pre-specified vulnerability rank (Table 1).
  - *Significance Threshold:* Permutation test $p < 0.05$ (5,000 permutations) after Holm family-wise error correction across H1–H3.

- **Hypothesis 2 (H2 - Cognitive Resilience at Matched Pathology):**
  Donors with high neuropathological burden but preserved cognition (Resilient: Braak $\ge$ IV, CDR $\le$ 0.5) exhibit distinct transcriptomic signatures compared to demented donors (Demented: Braak $\ge$ IV, CDR $\ge$ 1.0) when matched on age, sex, and neuropathological stage.
  - *Primary Metric:* Cross-validated classification AUC and differential expression effect sizes separating Resilient vs. Demented donors within cortical regions and pooled linear mixed-effects modeling.
  - *Decision Rule / Power Gate:* If the Resilient group has $N < 10$ donors, H2 is automatically re-designated as *Exploratory*, and this status change will be transparently documented in all reports.
  - *Significance Threshold:* Donor-bootstrap 95% CI of AUC excluding 0.50; Holm-adjusted $p < 0.05$.

- **Hypothesis 3 (H3 - External Replication & Generalizability):**
  Transcriptomic progression signatures and classifiers trained on a discovery cohort (GSE118553 / MSBB) generalize to independent lockbox validation cohorts (GSE132903, MSBB cross-platform) without parameter refitting or target cohort leakage.
  - *Primary Metric:* External AUC and rank correlation on untouched lockbox cohorts.
  - *Significance Threshold:* External AUC 95% bootstrap CI lower bound $> 0.50$; Holm-adjusted $p < 0.05$.

---

## 2. A-Priori Regional Vulnerability Ranking (for H1)

Based on established neuropathological progression sequences (Braak & Braak 1991, Thal et al. 2002, Wang et al. 2016), the 19 Mount Sinai Brain Bank (MSBB / GSE84422) regions and GSE118553 regions are ordered a-priori as follows:

| Rank Tier | Stage Category | Anatomical Brain Regions (MSBB / GSE118553) | Expected Severity |
|---|---|---|---|
| **Tier 1 (Earliest)** | Transentorhinal / Entorhinal (Braak I–II) | BM28 / BM34 (Entorhinal cortex, Parahippocampal gyrus), Hippocampus (CA1/Subiculum) | Highest early transcriptomic disruption |
| **Tier 2 (Limbic)** | Limbic / Temporal (Braak III–IV) | Amygdala, BM36 (Parahippocampal), BM38 (Temporal pole), BM20 (Inferior temporal), BM21/22 (Superior/Middle temporal) | Intermediate-to-high transcriptomic disruption |
| **Tier 3 (Neocortical Association)** | Isocortical / Polymodal Association (Braak V–VI) | BM9/46 (Dorsolateral Prefrontal Cortex), BM10 (Anterior Prefrontal), BM7/39/40 (Inferior/Superior Parietal), BM24/32 (Anterior Cingulate) | Late neocortical disruption |
| **Tier 4 (Primary Sensory/Motor)** | Primary Cortical Areas (Sparing until end-stage) | BM4 (Primary Motor Cortex), BM17 (Primary Visual Cortex), BM41/42 (Primary Auditory Cortex) | Minimal early disruption / delayed vulnerability |
| **Tier 5 (Subcortical / Cerebellar)** | Subcortical & Cerebellar | Caudate nucleus, Putamen, Cerebellar cortex / Cerebellum | Distinct non-tau pathways / Relative resistance |

*Note:* This ranking is locked prior to computing any regional correlation statistics on GSE84422 or GSE118553.

---

## 3. Data Cohorts and Allocation

1. **Discovery Cohort 1: GSE118553 (Patel et al.)**
   - Regions: Entorhinal cortex (EC), Hippocampus (HIP), Superior Frontal Gyrus (SFG), Cerebellum (CRB).
   - Clinical Groups: Control, Asymptomatic AD (AsymAD), Alzheimer's Disease (AD).
   - Platform: Illumina HumanHT-12 V4.0.

2. **Primary Pathology/Resilience Cohort: GSE84422 (Mount Sinai Brain Bank - MSBB)**
   - Regions: 19 distinct post-mortem brain regions (1,053 samples, 125 donors).
   - Phenotypes: CDR (0 to 5), Braak score (0 to VI), CERAD score, Plaque/Tangle metrics.
   - Platform: Affymetrix HG-U133A / HG-U133B.

3. **Lockbox Independent Validation Cohort: GSE132903 (Piras et al.)**
   - Region: Middle Temporal Gyrus (MTG).
   - Platform: Illumina HumanHT-12 V4.0 (same platform as GSE118553).
   - Status: STRICT LOCKBOX. Never used for feature selection, tuning, or training.

---

## 4. Methodological & Statistical Guardrails

1. **Unit of Independence:**
   - The biological donor is the strict unit of splitting and cross-validation.
   - Donor overlap between training and validation/test folds is strictly zero ($N_{overlap} = 0$).

2. **In-Fold Transformation & Feature Selection:**
   - Highly variable gene (HVG) selection, probe-to-gene summarization, scaling, imputation, dimensionality reduction (PCA/VAE), and hyperparameter tuning MUST occur strictly within each cross-validation fold.
   - No dataset-wide scaling or unsupervised pre-fitting on pooled data.

3. **Uncertainty Quantification:**
   - 95% Confidence Intervals (CIs) calculated via donor-level cluster bootstrap ($B = 2,000$ resamples).
   - Permutation testing ($N_{perm} \ge 5,000$) with empirical p-values calculated as $p = \frac{\sum I(T_{perm} \ge T_{obs}) + 1}{N_{perm} + 1}$.

4. **Covariate Adjustment & Sensitivity:**
   - Models adjust for age, biological sex, post-mortem interval (PMI), and RNA quality (RIN/pH) where recorded.
   - Age-matched sensitivity analyses (donors $\ge 65$ and 1:1 matching) required for all headline claims.

5. **Automated Verification:**
   - Test suite (`pytest tests/`) validating zero donor leakage, raw file checksums, and absence of synthetic/random data injectors.
   - Direct pipeline link from `results/*.json` to LaTeX macros (`paper/numbers.tex`). No manual statistic transcription.

---

## 5. Exploratory Analyses (Tiered)

All analyses outside H1–H3 are explicitly designated as *Exploratory* and will be cataloged in `ledger.csv`:
- **E1 (Leakage Taxonomy):** Systematic quantification of optimism bias across 4 leakage modes (donor leakage, split-wide feature selection, pooled normalization, test-set hyperparameter tuning).
- **E2 (Cell-Type Deconvolution):** BRETIGEA/marker-based cell-type proportion adjustment (neuronal, astrocytic, microglial, oligodendrocytic, endothelial) to decouple neurodegeneration from intrinsic cell-state changes.
- **E3 (Variance Partitioning):** Linear mixed modeling estimating donor-level systemic variance vs. region-specific variance.
- **E4 (Dissociation Analysis):** Independent identification of CDR-associated genes controlling for Braak vs. Braak-associated genes controlling for CDR.
- **E5 (Biological Anchoring):** Preranked GSEA against MSigDB Hallmarks, AMP-AD consensus co-expression modules (Wan et al. 2020), and AD GWAS risk genes (Bellenguez et al. 2022).
- **E6–E10:** Transcriptomic brain age acceleration, cross-region transfer matrix, sparse stability-selected gene panel, age-matched sensitivity reruns, and cVAE orthogonalization benchmark.

---

## 6. Pre-Registration Lock & Hash
- **Signed:** Automated Pre-Registration Commit
- **Execution Policy:** Any deviations from this protocol will be explicitly noted in the Supplementary Materials and logged in `ledger.csv`.
