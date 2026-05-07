# ADR 0001: generalisation_gap Sign Convention — ein − eout

**Date**: 2026-05-06  
**Status**: Accepted

## Context

The constitution (§V) defines `generalisation_gap = eout − ein` (negative = overfitting). The actual code in `MLTrainer` and `DLTrainer` logs `generalisation_gap = ein − eout` (positive = overfitting). Historical MLflow runs in `crash-severity-ml` and `crash-severity-dl` use the code convention.

The discrepancy surfaced during the GMM feature design (003-gmm-classifier) when T011 needed an explicit sign for the new trainer.

## Decision

All trainers — existing (`MLTrainer`, `DLTrainer`) and new (`GMMTrainer`) — log:

```
generalisation_gap = ein_macro_f1 − eout_macro_f1
```

Positive values indicate overfitting. Negative values indicate underfitting.

## Rationale

Changing the sign convention would make new GMM runs incomparable to all existing MLflow history. The constitution wording will be corrected separately (patch amendment, no behaviour change). Cross-trainer comparability in the MLflow UI outweighs strict adherence to the constitution's current wording.

## Consequences

- All three trainers are consistent in the MLflow UI
- The constitution §V wording must be updated in the next amendment to read `ein − eout`
- Any dashboard or alert that reads `generalisation_gap < 0` as "overfitting" is wrong — must flip to `> 0`
