"""Standard forecasting accuracy metrics."""
import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — handles zeros better than MAPE."""
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom != 0
    return float(np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)
