# Data Pipeline

## Overview

Raw sales data flows through three stages:

1. **Ingestion** — CSV/API → pandas DataFrame
2. **Cleaning** — outlier removal, missing value imputation
3. **Feature Engineering** — lag features, rolling averages, seasonality flags

## Supported Data Sources

| Source | Format | Update Frequency |
|--------|--------|------------------|
| Internal ERP | CSV export | Daily |
| Google Trends | API | Weekly |
| POS System | JSON API | Real-time |

## Quality Checks

- Missing value ratio < 5%
- No duplicate timestamps
- Value range within 3 IQR of historical median
