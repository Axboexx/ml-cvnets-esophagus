# ruff: noqa: ISC004 -- exact path fixtures are split only for readability.

import numpy as np
import pytest

import scripts.plot_f_model_roc_curves as roc_plot
from scripts.statistical_tests.delong import delong_estimates

EXPECTED_MODEL_PATHS = {
    "F1": (
        "F1/f1_tongji82_min4cm_4cm_nomargin_seed42/validation_predictions.csv",
        "F1/F1_external/f1_youyi_external_best/external_eval_predictions.csv",
    ),
    "F2a": (
        "F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42/"
        "validation_predictions.csv",
        "F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    "F2b": (
        "F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42/"
        "validation_predictions.csv",
        "F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_exterl/"
        "external_eval_predictions.csv",
    ),
    "E+L": (
        "F3_esophagus_liver/"
        "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_liver/"
        "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    "E+S": (
        "F3_esophagus_spleen/"
        "f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_spleen/"
        "f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    "E+PVS": (
        "F3_esophagus_portal_venous_system/"
        "f3_esophagus_portal_venous_system_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_portal_venous_system/"
        "f3_esophagus_portal_venous_system_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    "E+L+S": (
        "F3_esophagus_liver_spleen/"
        "f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_liver_spleen/"
        "f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    "E+L+PVS": (
        "F3_esophagus_liver_portal/"
        "f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_liver_portal/"
        "f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    "E+S+PVS": (
        "F3_esophagus_spleen_portal_venous_system/"
        "f3_esophagus_spleen_portal_venous_system_tongji82_min4cm_4cm_"
        "nomargin_seed42/validation_predictions.csv",
        "F3_esophagus_spleen_portal_venous_system/"
        "f3_esophagus_spleen_portal_venous_system_tongji82_min4cm_4cm_"
        "nomargin_seed42_youyi_external/external_eval_predictions.csv",
    ),
    "E+L+S+PVS": (
        "F3_esophagus_liver_spleen_portal/"
        "f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42/"
        "validation_predictions.csv",
        "F3_esophagus_liver_spleen_portal/"
        "f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
}

EXPECTED_FIGURES = [
    ("validation", ("F1", "F2a", "F2b")),
    ("external", ("F1", "F2a", "F2b")),
    (
        "validation",
        ("F2b", "E+L", "E+S", "E+PVS", "E+L+S", "E+L+PVS", "E+S+PVS", "E+L+S+PVS"),
    ),
    (
        "external",
        ("F2b", "E+L", "E+S", "E+PVS", "E+L+S", "E+L+PVS", "E+S+PVS", "E+L+S+PVS"),
    ),
]


def test_parse_epoch_overrides_accepts_trimmed_model_epoch_pairs() -> None:
    assert roc_plot._parse_epoch_overrides([" F1=0 ", "F2b=12"]) == {
        "F1": 0,
        "F2b": 12,
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["F1"], "must use MODEL=EPOCH"),
        (["unknown=1"], "unknown model"),
        (["F1=1", "F1=2"], "duplicate"),
        (["F1=1.0"], "invalid epoch"),
        (["F1=-1"], "non-negative integer"),
        (["F1=01"], "non-negative integer"),
    ],
)
def test_parse_epoch_overrides_rejects_ambiguous_values(
    values: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        roc_plot._parse_epoch_overrides(values)


def test_empirical_roc_points_have_the_same_auc_as_delong() -> None:
    targets = np.asarray([1, 0, 1, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float64)

    false_positive_rate, true_positive_rate = roc_plot._roc_curve(targets, scores)
    auc_from_points = np.sum(
        np.diff(false_positive_rate)
        * (true_positive_rate[:-1] + true_positive_rate[1:])
        / 2.0
    )
    auc_from_delong = delong_estimates(targets, [scores])[0].auc

    np.testing.assert_allclose(false_positive_rate, [0.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(true_positive_rate, [0.0, 0.5, 1.0, 1.0])
    assert auc_from_points == pytest.approx(0.875)
    assert auc_from_points == pytest.approx(auc_from_delong)


def test_four_figure_groups_resolve_expected_model_paths(monkeypatch, tmp_path) -> None:
    captured: list[dict[str, object]] = []

    def fake_resolve_model_inputs(**kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(roc_plot, "resolve_model_inputs", fake_resolve_model_inputs)
    epoch_overrides = {name: index for index, name in enumerate(EXPECTED_MODEL_PATHS)}

    for figure_spec in roc_plot.FIGURE_SPECS:
        assert roc_plot._model_inputs(tmp_path, figure_spec, epoch_overrides) == []

    assert [
        (spec.analysis_set, spec.models) for spec in roc_plot.FIGURE_SPECS
    ] == EXPECTED_FIGURES

    for (analysis_set, models), call in zip(EXPECTED_FIGURES, captured):
        path_index = 0 if analysis_set == "validation" else 1
        expected_paths = [
            f"{name}={tmp_path / EXPECTED_MODEL_PATHS[name][path_index]}"
            for name in models
        ]
        assert call["model_values"] == expected_paths
        assert call["metrics_values"] == []
        assert call["epoch_values"] == (
            [f"{name}={epoch_overrides[name]}" for name in models]
            if analysis_set == "validation"
            else []
        )


def test_missing_training_metrics_explains_epoch_override(
    monkeypatch, tmp_path
) -> None:
    model_specs = (
        roc_plot.ModelSpec("F1", "run_f1", "external_f1.csv"),
        roc_plot.ModelSpec("F2a", "run_f2a", "external_f2a.csv"),
    )
    monkeypatch.setattr(roc_plot, "MODEL_SPECS", model_specs)
    figure_spec = roc_plot.FigureSpec(
        "test_validation_roc", "validation", ("F1", "F2a")
    )
    for spec in model_specs:
        run_dir = tmp_path / spec.validation_run
        run_dir.mkdir()
        (run_dir / "validation_predictions.csv").touch()

    model_inputs = roc_plot._model_inputs(tmp_path, figure_spec, {"F1": 4, "F2a": 7})

    assert [(item.name, item.epoch) for item in model_inputs] == [
        ("F1", 4),
        ("F2a", 7),
    ]
    assert all(item.epoch_selection == "manual_override" for item in model_inputs)
    assert all(item.metrics_path is None for item in model_inputs)

    with pytest.raises(
        FileNotFoundError,
        match=r"training metrics not found:.*Supply the audited epoch with "
        r"--epoch MODEL=EPOCH\.",
    ):
        roc_plot._model_inputs(tmp_path, figure_spec, {})
