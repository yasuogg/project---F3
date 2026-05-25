"""Smoke tests for evaluation metrics."""
import random
from rlwa.eval.metrics import paired_bootstrap_pvalue


def test_paired_bootstrap_identical_returns_high_pvalue():
    rng = random.Random(0)
    xs = [rng.random() for _ in range(50)]
    diff, p = paired_bootstrap_pvalue(xs, list(xs), n_boot=500)
    assert abs(diff) < 1e-9
    assert p > 0.05  # should NOT reject under identical samples


def test_paired_bootstrap_strong_effect_significant():
    a = [1.0] * 30
    b = [0.0] * 30
    diff, p = paired_bootstrap_pvalue(a, b, n_boot=500)
    assert diff > 0.5
    assert p < 0.05
