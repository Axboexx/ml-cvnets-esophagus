import math

import numpy as np
import pytest

from scripts.statistical_tests.delong import (
    delong_auc_covariance,
    delong_estimates,
    pairwise_delong_tests,
)


def test_delong_matches_known_auc_covariance_and_p_value() -> None:
    targets = [1, 1, 1, 0, 0, 0]
    scores = [
        [0.9, 0.6, 0.4, 0.8, 0.5, 0.2],
        [0.95, 0.85, 0.55, 0.75, 0.65, 0.45],
    ]

    aucs, covariance = delong_auc_covariance(targets, scores)
    comparison = pairwise_delong_tests(targets, scores)[0]

    assert aucs == pytest.approx([2.0 / 3.0, 7.0 / 9.0])
    np.testing.assert_allclose(
        covariance,
        [[2.0 / 27.0, 1.0 / 18.0], [1.0 / 18.0, 5.0 / 81.0]],
    )
    assert comparison.auc_difference == pytest.approx(-1.0 / 9.0)
    assert comparison.z_statistic == pytest.approx(-1.0 / math.sqrt(2.0))
    assert comparison.p_value == pytest.approx(0.4795001221869535)


def test_delong_ties_equal_mann_whitney_auc() -> None:
    targets = [1, 1, 0, 0]
    scores = [[0.8, 0.5, 0.5, 0.2]]

    estimate = delong_estimates(targets, scores)[0]

    assert estimate.auc == pytest.approx(0.875)
    assert 0.0 <= estimate.ci_lower <= estimate.auc <= estimate.ci_upper <= 1.0


def test_identical_models_have_p_value_one() -> None:
    targets = [1, 1, 0, 0]
    scores = [0.9, 0.7, 0.4, 0.2]

    result = pairwise_delong_tests(targets, [scores, scores])[0]

    assert result.auc_difference == 0.0
    assert result.standard_error == 0.0
    assert result.z_statistic == 0.0
    assert result.p_value == 1.0


@pytest.mark.parametrize(
    ("targets", "scores", "message"),
    [
        ([1, 1, 1, 0], [[0.9, 0.8, 0.7, 0.1]], "at least 2 cases per class"),
        ([1, 1, 0, 0], [[0.9, np.nan, 0.2, 0.1]], "finite"),
        ([1, 2, 0, 0], [[0.9, 0.8, 0.2, 0.1]], "only 0 and 1"),
        ([1, 0.5, 0, 0], [[0.9, 0.8, 0.2, 0.1]], "only 0 and 1"),
    ],
)
def test_delong_rejects_invalid_inputs(targets, scores, message) -> None:
    with pytest.raises(ValueError, match=message):
        delong_auc_covariance(targets, scores)
