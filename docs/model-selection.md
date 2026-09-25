# Model Selection Guide

## Available Models

| Model | Best For | Training Time | Accuracy |
|-------|----------|---------------|----------|
| ARIMA | Stationary series | Fast | Good |
| Prophet | Strong seasonality | Medium | Good |
| LSTM | Complex patterns | Slow | Best |
| XGBoost | Feature-rich data | Medium | Great |

## Recommendation

Start with Prophet for quick baseline, then try XGBoost with
engineered features. Use LSTM only if data volume > 1000 points.
