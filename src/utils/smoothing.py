"""Time series smoothing utilities."""
import numpy as np


def simple_moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Compute SMA with given window size."""
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def exponential_moving_average(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Compute EMA — higher alpha = more weight on recent values."""
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def double_exponential_smoothing(values: np.ndarray, alpha: float = 0.3, beta: float = 0.1) -> np.ndarray:
    """Holt's method — captures level and trend."""
    n = len(values)
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = values[0]
    trend[0] = values[1] - values[0] if n > 1 else 0

    for i in range(1, n):
        level[i] = alpha * values[i] + (1 - alpha) * (level[i-1] + trend[i-1])
        trend[i] = beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1]

    return level + trend
