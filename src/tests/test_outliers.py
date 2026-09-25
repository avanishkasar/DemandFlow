"""Tests for outlier detection."""
import numpy as np
from utils.outliers import detect_outliers, clip_outliers


def test_no_outliers_in_uniform_data():
    data = np.array([10, 11, 12, 10, 11, 13, 10, 12])
    mask = detect_outliers(data)
    assert not mask.any()


def test_detects_extreme_value():
    data = np.array([10, 11, 12, 10, 11, 100, 10, 12])
    mask = detect_outliers(data)
    assert mask[5]  # the 100 should be flagged


def test_clip_replaces_outliers():
    data = np.array([10, 11, 12, 10, 11, 100, 10, 12])
    clean = clip_outliers(data)
    assert clean[5] < 100
    assert clean[5] >= 10
