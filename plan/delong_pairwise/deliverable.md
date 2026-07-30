# Deliverable

## Entry point

`scripts/compare_validation_auc_delong.py`

## Required files

For F1, F2a, and F2b, provide:

- `validation_predictions.csv`
- sibling `training_metrics.csv`

The metrics file may be omitted only when an explicit `--epoch MODEL=N` is
provided. The script requires identical `case_id` sets and targets across all
models and never silently takes an intersection.

## Server command template

```bash
cd /path/to/ml-cvnets

python3 scripts/compare_validation_auc_delong.py \
  --model F1=/work/results/F1/<F1_RUN>/validation_predictions.csv \
  --model F2a=/work/results/F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42/validation_predictions.csv \
  --model F2b=/work/results/F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42/validation_predictions.csv \
  --output-dir /work/results/delong_f1_f2a_f2b_validation_seed42
```

The output directory must not already exist.

## Outputs

- `model_auc.csv`: AUC, standard error, and 95% CI per model.
- `pairwise_delong.csv`: all pairwise AUC differences, 95% difference CIs,
  z-statistics, raw P values, and Holm-adjusted P values.
- `analysis_metadata.json`: input and implementation hashes, selected epochs,
  cohort hash, covariance matrix, and environment metadata.
- `README.md`: readable result summary and interpretation boundary.

## Verification

- 15 targeted synthetic tests passed.
- Ruff `E,F,I` checks passed.
- CLI help invocation passed.
