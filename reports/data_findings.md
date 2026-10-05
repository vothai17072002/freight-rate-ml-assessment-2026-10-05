# Freight-rate assessment data findings

This report is generated from the supplied 2025 assessment CSVs. Input files are read only; their SHA-256 hashes are recorded in `data_audit.json`.

Reproduce from the project root: `python -m src.data_audit --data-dir data --output-dir reports`.

## Dataset roles and coverage

| Input | Rows | Date coverage | Target |
|---|---:|---|---|
| train_test.csv | 48,000 | 2025-01-01 to 2025-10-31 | posted_rate supplied |
| validation.csv | 12,000 | 2025-11-01 to 2025-12-31 | posted_rate withheld |
| validation_predictions_template.csv | 12,000 | Uses validation IDs | predicted_rate blank |
| december_chart_inputs.csv | 31 | 2025-12-01 to 2025-12-31 | predicted_rate blank |

## Quality checks

| Input | Missing weight | Nonpositive weight | Missing market_index | Duplicate IDs | Exact duplicate rows |
|---|---:|---:|---:|---:|---:|
| train_test.csv | 300 | 292 | 374 | 0 | 0 |
| validation.csv | 165 | 145 | 249 | 0 | 0 |

Numeric parse failures: 0. Nonfinite numeric values: 0. Invalid ISO dates: 0.
Nonpositive shipment weight is treated as invalid for modelling. It should be handled explicitly as missing or through a documented policy; the audit does not change or take the absolute value of the source data.
Coordinate range checks only test mathematical latitude/longitude bounds. They do not establish that supplied coordinates match the real cities; no external coordinate correction is performed.

## Generalization and split implications

Training ends on 2025-10-31; external validation starts on 2025-11-01. Validation therefore measures prediction into a later period. Development should reserve a later training period for holdout validation and fit all imputers, encoders and derived-feature lookup tables only on the training side of each split.
Validation has 8 unseen cities: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego. 1,447 of 12,000 validation rows contain an unseen city.
Ordered lane means pickup to delivery; reverse direction is distinct. Validation has 736 unseen ordered lanes across 1,461 rows; 1,852 rows have an unseen lane and equipment combination.
These observations support temporal evaluation and explicit fallbacks for unseen categorical values. A random row split alone does not reproduce the supplied external-validation setting.

## Target distribution

| Statistic | posted_rate |
|---|---:|
| min | 57.22 |
| p01 | 327.17 |
| p05 | 599.74 |
| median | 2,030.76 |
| mean | 2,373.98 |
| p95 | 4,953.77 |
| p99 | 5,972.83 |
| max | 25,533.00 |

Extreme rates require review against distance and equipment; magnitude alone does not prove the target is erroneous.

## Numerical associations and feature provenance

| Feature | Comparison target | Paired rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| distance | posted_rate | 48,000 | 0.9085 | 0.9760 |
| weight | posted_rate | 47,700 | 0.0348 | 0.0417 |
| market_index | posted_rate | 47,626 | 0.0342 | 0.0315 |
| quote_signal | posted_rate | 48,000 | -0.0399 | -0.0399 |
| quote_signal | posted_rate / distance | 48,000 | 0.0493 | 0.0911 |

The `posted_rate / distance` diagnostic is computed from the supplied target only to examine training associations. It must never be supplied as a prediction-time feature.
Overall association can hide temporal instability. The monthly relationship between `quote_signal` and the rate per distance unit is:

| Month | Paired rows | Pearson | Spearman |
|---|---:|---:|---:|
| 2025-01 | 4,918 | 0.4467 | 0.9625 |
| 2025-02 | 4,337 | 0.4790 | 0.9650 |
| 2025-03 | 5,036 | 0.4858 | 0.9701 |
| 2025-04 | 4,819 | -0.4717 | -0.9634 |
| 2025-05 | 4,913 | -0.4592 | -0.9679 |
| 2025-06 | 4,783 | 0.4766 | 0.9681 |
| 2025-07 | 4,912 | -0.4683 | -0.9646 |
| 2025-08 | 4,759 | -0.0136 | -0.0224 |
| 2025-09 | 4,670 | 0.4814 | 0.9665 |
| 2025-10 | 4,853 | -0.4340 | -0.9593 |

The observed quote-signal relationship reverses direction between training months and is near zero in August. Its pooled correlation does not capture those changes. This supports temporal stress tests and an ablation excluding this field; external-validation labels are unavailable, so the relationship in November and December is not measurable here.
`quote_signal` has no provenance, measurement timing or computation definition in the supplied CSV headers. A strong within-month association with the target can come from a valid prior quote or from a target-derived signal. Correlation alone cannot distinguish these cases and does not confirm leakage. Any production-use claim depends on this field being available before the posted rate is known.
`market_index` likewise needs a prediction-time availability policy; its missingness and its absence from the December input should be handled without using future target information.

## Submission and December scenario

The prediction template ID set matches validation: True; row order matches: True. The template has 12,000 blank predictions.
The December scenario requests 31 daily predictions for Lexington to Fort Wayne, distance 360, Dry Van, weight 32000. The daily grid contains exactly 2025-12-01 through 2025-12-31 once each.
It omits these features present in validation: delivery_lat, delivery_lon, market_index, pickup_lat, pickup_lon, quote_signal. The inference pipeline must support these absent columns through documented training-only fallback estimates or an appropriate reduced-feature model.
Training contains 32 rows on the same ordered lane and 21 with the same equipment. Historical support does not supply the missing future market or quote signal.

## Monthly row distribution

| Month | Training rows | Validation rows |
|---|---:|---:|
| 2025-01 | 4,918 | 0 |
| 2025-02 | 4,337 | 0 |
| 2025-03 | 5,036 | 0 |
| 2025-04 | 4,819 | 0 |
| 2025-05 | 4,913 | 0 |
| 2025-06 | 4,783 | 0 |
| 2025-07 | 4,912 | 0 |
| 2025-08 | 4,759 | 0 |
| 2025-09 | 4,670 | 0 |
| 2025-10 | 4,853 | 0 |
| 2025-11 | 0 | 5,836 |
| 2025-12 | 0 | 6,164 |
