# Resume notes

## Bullets

- Classified 10 bearing fault conditions from accelerometer signals with a one-dimensional CNN at 0.9717 macro-F1 on an unseen motor load vs. 0.9668 for a spectral-feature random forest.
- Built FFT and envelope-spectrum features; the detected outer-race fault peak matched bearing-geometry BPFO within 0.19%.
- Predicted remaining useful life on NASA C-MAPSS engine telemetry with LightGBM and a one-dimensional CNN, reaching 15.11 RMSE (cycles) on the official FD001 test set.
- Assigned each CWRU recording to one motor-load split so cross-split sample overlap is 0 windows. The held-out load is 3 hp.

## Skills and tools

Python, SQL, DuckDB, NumPy, SciPy, scikit-learn, PyTorch, LightGBM, FFT, Hilbert envelope spectrum, bearing fault frequencies, one-dimensional CNN, random forest, time series, predictive maintenance.

## Summary

This project classifies drive-end bearing faults from public accelerometer recordings and predicts remaining useful life on NASA turbofan telemetry. Windows are split by motor load so neighboring samples cannot leak into the test set, and an envelope spectrum is checked against the outer-race fault frequency from the bearing geometry.

## GitHub description

Classify CWRU bearing faults from vibration spectra and a 1D CNN, on a turbofan RUL pipeline.
