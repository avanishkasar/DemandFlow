"""Rolling window calculations with NaN handling."""
import numpy as np


def rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        result[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return result
