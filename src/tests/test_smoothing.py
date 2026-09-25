"""Tests for smoothing utilities."""
import numpy as np
from utils.smoothing import simple_moving_average, exponential_moving_average


def test_sma_window_3():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = simple_moving_average(data, window=3)
    assert len(result) == 3
    assert result[0] == 2.0  # mean of [1, 2, 3]


def test_ema_reacts_to_changes():
    data = np.array([10.0] * 5 + [20.0] * 5)
    result = exponential_moving_average(data, alpha=0.5)
    assert result[-1] > result[0]
    assert result[-1] < 20.0  # shouldn't fully reach 20
