"""Feature scaling and normalization for model input."""
import numpy as np


class MinMaxScaler:
    """Simple min-max scaler that remembers fit parameters."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom[denom == 0] = 1  # avoid division by zero
        return (data - self.min_) / denom

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max_ - self.min_) + self.min_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)
