"""Date manipulation helpers for forecast windows."""
from datetime import datetime, timedelta


def get_date_range(start: str, periods: int, freq: str = "D") -> list[str]:
    """Generate a list of date strings from a start date."""
    dt = datetime.strptime(start, "%Y-%m-%d")
    delta = {"D": timedelta(days=1), "W": timedelta(weeks=1), "M": timedelta(days=30)}
    step = delta.get(freq, timedelta(days=1))
    return [(dt + step * i).strftime("%Y-%m-%d") for i in range(periods)]


def quarter_label(date_str: str) -> str:
    """Convert 2024-03-15 → Q1 2024."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"
