# Resume notes

## Bullets

- Predicted remaining useful life on NASA C-MAPSS engine telemetry with LightGBM and a one-dimensional CNN, reaching 15.11 RMSE (cycles) on the official FD001 test set.
- Designed an alert policy combining RUL forecasts with control-chart and isolation-forest anomaly scores that warned 34.50 cycles ahead for 100.00% of engines at 0.0000 false alerts per engine.
- On FD003, the same models reached 15.98 test RMSE (cycles), compared with 45.07 for a constant mean-RUL baseline.
- Built the SQL feature layer in DuckDB and a static operations dashboard for alert review.

## Skills and tools

Python, SQL, DuckDB, LightGBM, PyTorch, scikit-learn, pandas, statistics, anomaly detection, control charts, time series, predictive maintenance, matplotlib, Plotly.

## Summary

This project predicts how many cycles a turbofan engine has left using NASA C-MAPSS sensor streams. It builds rolling features in SQL, compares gradient boosting with a one-dimensional convolutional network, and checks an alert rule against the actual failure times.

## GitHub description

Predict remaining useful life and failure alerts from NASA C-MAPSS turbofan telemetry.
