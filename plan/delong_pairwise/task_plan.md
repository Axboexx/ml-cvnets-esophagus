# F1/F2a/F2b validation AUC pairwise DeLong plan

## Goal

Add a reproducible command-line analysis that compares F1, F2a, and F2b on
the same validation cases, reports per-model AUC with 95% confidence intervals,
and performs all three paired DeLong tests with Holm correction.

## Scope

- Read existing `validation_predictions.csv` files without modifying them.
- Resolve each model's final best validation epoch from `training_metrics.csv`,
  with an explicit epoch override as a fallback.
- Fail closed when cohorts, targets, stages, epochs, or probabilities are
  inconsistent.
- Write a compact CSV/JSON/Markdown result bundle to a new output directory.
- Add synthetic unit/integration tests; do not require access to server data.

## Phases

- [x] Inspect current prediction export schema and best-checkpoint bookkeeping.
- [x] Define statistical and CLI contracts.
- [x] Implement DeLong statistics and strict input alignment.
- [x] Add tests for calculations, epoch selection, and invalid inputs.
- [x] Run available verification and review the diff.
- [x] Document the server command template and required files.

## Decisions

- Unit of analysis: one `case_id` per validation case.
- Score: `positive_probability`; binary target: `target`.
- Partition: `stage=validation` and the selected best epoch for each model.
- Comparisons: F1–F2a, F1–F2b, and F2a–F2b.
- Inference: paired DeLong covariance, two-sided raw P value, and Holm-adjusted
  P value across the three planned comparisons.
- AUC confidence intervals: two-sided 95% normal-approximation DeLong intervals,
  clipped to `[0, 1]`.
- Scientific boundary: these are internal validation comparisons after
  validation-based model selection, so the report labels them exploratory and
  does not treat them as external generalization estimates.

## Error log

| Issue | Resolution |
|---|---|
| System Python lacks NumPy and pytest | Used an isolated `uv` environment with a temporary cache; no project dependencies were changed. |
| Initial package import failed under pytest | Added direct-script and package import paths, then reran tests. |
| Review found input-hash race and partial output risk | Added before/after input hash verification and atomic directory publication. |

## Verification

- Targeted pytest: `15 passed`.
- Ruff `E,F,I`: passed.
- Black formatting: applied to all added Python files.
- Direct CLI `--help`: passed.
