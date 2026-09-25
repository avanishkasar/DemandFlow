"""Central configuration for forecast models."""
from dataclasses import dataclass, field


@dataclass
class ForecastConfig:
    """Configuration for the demand forecasting pipeline."""
    horizon: int = 30
    confidence_interval: float = 0.95
    seasonality_mode: str = "multiplicative"
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    train_test_split: float = 0.8
    random_seed: int = 42
    features: list[str] = field(default_factory=lambda: [
        "day_of_week", "month", "is_holiday", "lag_7", "lag_30", "rolling_mean_7",
    ])


DEFAULT_CONFIG = ForecastConfig()
