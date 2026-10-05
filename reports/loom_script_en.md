# Suggested Loom walkthrough

Target: 2-3 minutes at a natural speaking pace. The spoken script below is about
350 words. Screen cues are not spoken. Review the code and numbers first, then
record this in your own words. This file is a script, not a recorded Loom video.

## 0:00-0:35 — Data audit

Show `reports/data_findings.md`, then the temporal split in the PDF.

> This solution predicts freight prices for twelve thousand November and
> December loads. The forty-eight thousand labeled loads cover January through
> October, so I used chronological validation to match the future-period task.
>
> The audit found missing and nonpositive weights, missing market values, and
> eight new cities in the final validation inputs. I treat invalid weights as
> missing, keep an indicator, and fit imputation values only on training data.
> I retain expensive target values because size alone does not prove an error.

## 0:35-1:10 — Features and model

Show `src/features.py`, `src/model.py`, and `reports/model_comparison.png`.

> A key finding was that quote signal changes its relationship with price
> between development months. Its provenance is also unspecified. I excluded
> it conservatively; that instability does not prove leakage.
>
> I compared a regularized linear model with CatBoost configurations. The
> selected CatBoost fits log rate per mile, then converts predictions back to
> dollars using distance. It captures nonlinear shipment and calendar effects
> and supports categorical locations. The primary model also uses coordinates
> and market index, assuming that index is available before pricing.

## 1:10-1:55 — Evaluation

Show `src/train.py` and page 4 of the PDF.

> I trained candidates on January through June and selected configurations on
> July and August. I froze those choices before evaluating September and
> October, then refitted on all labeled rows for submission.
>
> On the nine thousand five hundred twenty-three holdout loads, the primary
> model achieves six hundred thirty-eight dollars RMSE, one hundred twenty-two
> dollars MAE, and five point one one percent weighted absolute percentage
> error. The baseline RMSE is six hundred seventy-one dollars. A separate
> controlled cold-city test provides supporting generalization evidence, but
> it does not measure accuracy on the actual new validation cities.

## 1:55-2:40 — Code and outputs

Show `src.predict.py`, `validation_predictions.csv`, page 5 of the PDF, and README
commands.

> Training and inference share the feature function. The model wrapper stores
> training-only preprocessing. Prediction maps by load identifier and preserves
> template order.
>
> The December chart inputs omit coordinates and market signals, so I trained
> a separate six-input model. Its curve is a predictive scenario, not observed
> December prices. The original scorer accepts all twelve thousand predictions
> and thirty-one December rows, then generates this chart. It does not calculate
> accuracy on the withheld final labels.
>
> The repository includes pinned dependencies, recorded metrics, tests and a
> single command to reproduce the submission.

## After recording

Check that the video is 2-3 minutes and the reviewer can access the link. Submit
the actual Loom link with the repository URL, prediction CSV and PDF. If you use
a private GitHub repository, grant reviewer access separately.
