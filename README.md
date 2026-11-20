# Biochar rate responses and the value of local measurements across global experiments

This repository contains the core analysis code for the BiocharDS study of within-experiment rate responses and prediction in held-out studies. It includes data preparation, covariate benchmarks, trial-anchored contrast learning (TRACE), matched-budget mixed-effects comparisons, residual interval diagnostics and treatment selection.

## Input

Download the [BiocharDS version 1.0 workbook](https://ndownloader.figshare.com/files/43579383) and save it as `BiocharDS_V1.0.xlsx` in the repository root. The analysis reads `Sheet1`. The source workbook has SHA-256 `483cd75782ce83937b3228e55a0e7e743d8f4fc37b54ea6b564bb0f44022fb20`. The original data are available from [Figshare](https://doi.org/10.6084/m9.figshare.24781737.v1) and described by [Li et al. in Scientific Data](https://doi.org/10.1038/s41597-023-02867-9).

## Run

Use Python 3.12 and install the listed packages in a clean environment. Run the following from the repository root.

```shell
python -m pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_structure_diagnostics.py
python scripts/14_cobenefit.py
```

The cleaning step produces 2,438 treatment–control records from 367 studies and 891 experiment identifiers. Endpoint-specific analyses use the rows with an observed response. In `tables/T14_cobenefit.csv`, the joint improvement frequency is approximately 63% for yield and soil organic carbon and 38% for yield and nitrous oxide. These frequencies refer to matched higher-versus-lower-rate contrasts.

## Analyses

After cleaning, run each numbered script from the root as above. The prediction benchmarks fit many models in parallel and are computationally intensive. Scripts write intermediate predictions under `results/` and summary CSVs under `tables/`.

| Script | Purpose and prerequisites |
| --- | --- |
| 03 | Fixed-configuration benchmarks for random-row and study-held-out splits. |
| 04 | Fixed-configuration anchored models and their comparators. |
| 05 | Pooled and within-experiment rate associations and exploratory subgroup estimates. |
| 10 | Study-grouped nested tuning of anchored learners and calibration baselines. Writes ordered anchor–target predictions, selected settings and study-bootstrap comparisons. |
| 11 | Nested tuning of covariate-only benchmarks under both split schemes. |
| 13 | After 10, calculates empirical residual widths and coverage across validation folds. Other-fold residuals may come from models that used focal-fold studies during fitting. |
| 14 | Joint endpoint gains across rate contrasts sharing experiment, treatment code, biochar type and pyrolysis temperature. Runs directly after 01. |
| 16 | After 10, compares predictive skill and treatment-selection accuracy and assesses stability. Its time split uses publication year. |
| 17 | After 10, fits same-budget BLUP and Mundlak comparators, matches anchor–target pairs, and summarises the incremental skill associated with learned contrasts. |
| 18 | After 05, cross-fits within-experiment and correlated-effects DML specifications by study. The `within_fe_dml` identifier from script 05 refers to an earlier linear covariate adjustment. |
| 19 | Checks how rate estimates change after matching the observed control value within experiment. To also run the fixed-model prediction sensitivity, run 04 first, then `python scripts/19_strict_unit.py predict`. |

The shared modules `common.py` and `trace_core.py` define experiment grouping, fold assignment, metrics, ordered pair construction and contrast features. Cross-study tests with an anchor deliberately reveal one measurement from an experiment in the withheld study. The remaining responses serve as prediction targets. Tables T10, T16 and T17 distinguish the benefit of the measured anchor, the added predictive skill of learned contrasts and the accuracy of selecting among observed treatments.
