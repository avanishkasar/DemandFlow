"""Indian market holiday detection."""
from datetime import date

FIXED_HOLIDAYS = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (12, 25): "Christmas",
}


def is_holiday(d: date) -> bool:
    return (d.month, d.day) in FIXED_HOLIDAYS


def get_holiday_name(d: date) -> str | None:
    return FIXED_HOLIDAYS.get((d.month, d.day))
