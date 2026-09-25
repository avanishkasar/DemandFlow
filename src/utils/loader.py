"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True
