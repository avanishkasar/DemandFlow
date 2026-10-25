"""Seasonal decomposition for demand signals."""
import numpy as np


def estimate_seasonality(values: np.ndarray, period: int = 7) -> np.ndarray:
    """Estimate seasonal component by averaging over periods."""
    n = len(values)
    seasonal = np.zeros(n)
    for i in range(period):
        indices = list(range(i, n, period))
        mean_val = np.mean(values[indices])
        for idx in indices:
            seasonal[idx] = mean_val
    # Normalize so seasonal sums to zero per period
    overall_mean = np.mean(seasonal[:period])
    seasonal -= overall_mean
    return seasonal


def deseasonalize(values: np.ndarray, period: int = 7) -> np.ndarray:
    """Remove seasonal component from time series."""
    seasonal = estimate_seasonality(values, period)
    return values - seasonal
