# M2 Error Analysis - frozen FD001 model (post-hoc official test)

Model: `gru_w45_huber` (M2 CV-selected, 85 development engines). Official test: 100 engines. Total NASA = 6700.59.

**Terminology correction (M2_REPAIR_PLAN.md R1/R2):** official test trajectories are truncated before failure; `cycle.max()` = observed history length, NOT failure lifetime. `implied_failure_cycle = observed_cycles + true_rul`. `error = prediction - true_rul` (positive = overprediction).

## Profile

| split | group | count | mean error | mean abs error | RMSE | MAE | NASA sum | NASA % of total |
|---|---|---|---|---|---|---|---|---|
| observed_history | padded_observed_lt_window | 4 | 45.874 | 45.874 | 47.073 | 45.874 | 721.7 | 10.77 |
| observed_history | observed_45_127 | 40 | 17.426 | 22.843 | 29.678 | 22.843 | 5865.3 | 87.53 |
| observed_history | observed_ge_128 | 56 | -4.021 | 8.289 | 12.671 | 8.289 | 113.5 | 1.69 |
| implied_lifetime | implied_lifetime_lt_128 | 0 | nan | nan | nan | nan | 0.0 | empty |
| implied_lifetime | implied_lifetime_ge_128 | 100 | 6.554 | 15.614 | 23.041 | 15.614 | 6700.6 | 100.0 |
| true_rul_bin | true_rul_0_29 | 25 | 1.213 | 4.258 | 5.703 | 4.258 | 15.6 | 0.23 |
| true_rul_bin | true_rul_30_59 | 14 | 1.768 | 6.876 | 11.421 | 6.876 | 44.5 | 0.66 |
| true_rul_bin | true_rul_60_99 | 27 | 10.972 | 19.546 | 27.436 | 19.546 | 3339.8 | 49.84 |
| true_rul_bin | true_rul_ge_100 | 34 | 8.943 | 24.44 | 29.766 | 24.44 | 3300.7 | 49.26 |
| padded_vs_full | padded_observed_lt_45 | 4 | 45.874 | 45.874 | 47.073 | 45.874 | 721.7 | 10.77 |
| padded_vs_full | full_observed_ge_45 | 96 | 4.915 | 14.353 | 21.463 | 14.353 | 5978.9 | 89.23 |

## What the M1 claim becomes

M1 reported that 'engines with lifetime < 128 carry 99.8% of the NASA penalty'. Under the corrected definition (implied failure lifetime = observed_cycles + true_rul), the `implied_lifetime_lt_128` group is **empty on the official test** (100/100 engines): every official engine has implied failure cycle >= 128. The M1 finding was really about OBSERVED HISTORY LENGTH (44 engines with cycle.max() < 128); that quantity is a trajectory-truncation artifact, not a lifetime. The observed-history rows above supersede it.

Observed-history split of the old claim: 40 engines in observed 45-127, 56 in observed >= 128, 4 padded (< window).

## Notes

- `padded_observed_lt_window`: engines whose observed history is shorter than the model window (45); their windows are left-padded in the shared representation. This is expected input for the model, not out-of-distribution.
- Only 4 of 100 official-test engines have observed history < 45; small-sample caution applies to that subgroup.