# Illustrative code: core methods of the reservoir-forecasting study

This repository contains **illustrative code snippets** for the core methods described in the paper *Point Accuracy Does Not Guarantee Low-Storage Event Detection: Medium-Range Reservoir Storage Forecasting in Jeonbuk*:

- `src/eval.py`: event-level evaluation (event segments, overlap-based hits, precision and recall) and the metrics shared by the analyses.
- `scripts/run_baselines.py`: the pooled gradient-boosting (HistGradientBoosting) model with the frozen protocol-v2 hyperparameters.
- `scripts/run_conformal_events.py`: conformal calibration of prediction intervals and the event-level evaluation of their warnings.

These files show how the methods were implemented.

- License: MIT (see `LICENSE`). Copyright holder: [Author names to be confirmed]
