"""Outlier detection for demand time series."""
import numpy as np
from typing import Literal


def detect_outliers(
    values: np.ndarray,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> np.ndarray:
    """Return boolean mask where True indicates an outlier."""
    if method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (values < lower) | (values > upper)
    else:
        mean, std = np.mean(values), np.std(values)
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
        return z_scores > threshold


def clip_outliers(values: np.ndarray, method: str = "iqr") -> np.ndarray:
    """Replace outliers with boundary values."""
    mask = detect_outliers(values, method=method)
    clean = values.copy()
    if mask.any():
        valid = values[~mask]
        clean[mask] = np.clip(values[mask], valid.min(), valid.max())
    return clean
