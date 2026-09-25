"""Tests for forecasting metrics."""
import numpy as np
import pytest
from utils.metrics import mape, rmse, mae, smape


class TestMAPE:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        assert mape(actual, actual) == 0.0

    def test_known_error(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        result = mape(actual, predicted)
        assert 5.0 < result < 15.0


class TestRMSE:
    def test_perfect_forecast(self):
        actual = np.array([1, 2, 3])
        assert rmse(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([3.0, -0.5, 2.0])
        predicted = np.array([2.5, 0.0, 2.1])
        assert 0.0 < rmse(actual, predicted) < 1.0


class TestSMAPE:
    def test_handles_zeros(self):
        actual = np.array([0.0, 100.0])
        predicted = np.array([0.0, 90.0])
        result = smape(actual, predicted)
        assert result >= 0
