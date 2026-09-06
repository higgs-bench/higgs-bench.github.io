# HIGGS Imbalance Benchmark

Do class-imbalance techniques actually improve classifier ranking on the HIGGS
boson dataset? Across 450 runs (7 models x 5 techniques x 3 imbalance ratios
x 5 seeds), **no technique produced a significant improvement, and most
produced significant harm.**

**Interactive results:** https://rudraindia.github.io/higgs-imbalance-benchmark/

## Headline results

Change in AUPRC vs an untreated baseline, same model and same test set.

| Technique | Version B (1:10) | Version C (1:50) |
|---|---|---|
| SMOTE | hurts 7/7 models, p=0.016, -0.067 | hurts 7/7, p=0.016, -0.035 |
| Random undersampling | hurts 7/7, p=0.016, -0.026 | hurts 7/7, p=0.016, -0.016 |
| Class weighting | hurts 6/7, -0.010 | hurts 7/7, p=0.016, -0.012 |
| Focal loss | null | null |

Across the 46 technique cells in the two imbalanced versions: **zero
significant improvements.** In the balanced control (Version A) 22 of 23
cells are null, as expected when there is no imbalance to correct.

Two methodological findings:

1.  **Single-seed benchmarks understate uncertainty by roughly 28x.** Median
   model-seed interval width is 0.0009; median test-set bootstrap interval
   width is 0.0250. At this test-set size model ranking is unresolvable — —
   20 to 22 of 30 cells overlap the best cell's confidence interval.
2. **Threshold calibration explains the apparent value of rebalancing.** At
   1:50, tuning the decision threshold on validation lifts baseline F1 from
   0.006 to 0.169. Techniques that look strong at a fixed 0.5 threshold lose
   that advantage once the threshold is tuned, while keeping their ranking cost.

## Protocol

- Fixed pool of 200k background + 200k signal drawn from 11M rows
- Stratified 60/20/20 train/val/test split; imbalance created by truncating the signal class
- Test sets stay imbalanced; Version C's splits are subsets of B's, which are subsets of A's
- Resampling applied to training data only, never to validation or test
- Decision threshold tuned on validation, never on test
- Primary metric AUPRC, always reported against the prevalence baseline
- 2000 stratified bootstrap resamples of the test set for intervals; paired resamples for deltas
- Sign test across the 7 models for technique-level conclusions

## Reproduce

    py -3.12 -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

    # set paths.raw_csv in configs/default.yaml
    python -m higgs_bench.data                                            # build splits, ~2 min
    pytest -q                                                             # 32 tests
    python scripts\run_sweep.py --workers 3                               # ~8 h
    python scripts\run_sweep.py --workers 3 --seeds 0 --save-predictions  # ~2 h
    python scripts\compile_results.py
    python scripts\bootstrap_stats.py
    python scripts\make_figures.py

Dataset: https://archive.ics.uci.edu/dataset/280/higgs (7.5 GB, not included in this repo)

## Note on signal significance

Z = TP/sqrt(TP+FP) is reported for continuity with prior work but is
**scale-dependent**: it grows as sqrt(test-set size) with no change in
classifier quality. It is not a physics discovery significance and is not
compared against Z >= 5 anywhere in this work. See `src/higgs_bench/metrics.py`.

## Repository layout

    configs/default.yaml        experiment configuration
    src/higgs_bench/data.py     dataset construction and split loading
    src/higgs_bench/models.py   model factory (7 estimators)
    src/higgs_bench/techniques.py  imbalance techniques and focal loss
    src/higgs_bench/metrics.py  evaluation metrics and threshold tuning
    src/higgs_bench/runner.py   single experiment cell
    scripts/                    sweep, compilation, bootstrap, figures
    tests/                      32 tests including a finite-difference
                                check on the focal-loss gradient



## Rebuilding the site

    cd web
    npm install
    npm run dev      # local development
    npm run build    # writes to ../docs, which GitHub Pages serves




## Authors

1. Kritika Sharma : https://github.com/KRITIKA967
2. Rudra Raj : https://github.com/RUDRAIndia