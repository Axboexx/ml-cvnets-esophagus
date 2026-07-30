# Investigation notes

## Existing output schema

`validation_predictions.csv` contains:

- `epoch`
- `stage`
- `sample_index`
- `case_id`
- `target`
- `prediction`
- `positive_probability`
- `negative_probability`
- `positive_logit`
- `negative_logit`
- `is_correct`

The writer preserves multiple `(epoch, stage)` partitions in one file and
normalizes distributed-sampler duplicates before writing.

## Best epoch

`training_metrics.csv` contains `epoch`, `is_best`, and `is_ema_best`.
Checkpoint replacement uses `>=` or `<=`, so ties replace the previous best.
Therefore the matching non-EMA `checkpoint_best.pt` epoch is the last row whose
`is_best` value is true.

## Required server inputs

For each of F1, F2a, and F2b:

1. `validation_predictions.csv`
2. `training_metrics.csv`, unless an explicit best epoch is supplied

All three prediction files must describe the identical validation cohort and
identical target for every `case_id`.

