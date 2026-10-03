import pytest

from irag.tools.paper_statistics import (
    exact_sign_flip_p,
    holm_adjust,
    summarize,
)


def test_summarize_uses_sample_sd_and_student_t_interval():
    result = summarize([float(value) for value in range(1, 11)])

    assert result["mean"] == 5.5
    assert result["sample_sd"] == pytest.approx(3.0276503541)
    assert result["ci95_low"] == pytest.approx(3.334149, abs=1e-6)
    assert result["ci95_high"] == pytest.approx(7.665851, abs=1e-6)


def test_exact_sign_flip_p_is_two_sided_and_exact():
    assert exact_sign_flip_p([1.0] * 10) == pytest.approx(2 / 1024)
    assert exact_sign_flip_p([1.0, -1.0]) == 1.0


def test_holm_adjust_is_monotonic_in_sorted_p_values():
    assert holm_adjust([0.01, 0.04, 0.03, 0.20]) == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )
