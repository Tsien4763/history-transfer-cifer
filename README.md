# When Does History Transfer? (CIFEr Core)

[中文说明](README_ZH.md)

Under a temporal regime shift, when does reusing past forecast errors from a related series improve accuracy, and when should a model fall back to its own local residual?

This repository contains the CIFEr core implementation of *When Does History Transfer?* (IEEE SSCI 2027 full paper; NeurIPS 2026 FMTS Workshop accepted). Rather than pooling all historical residuals unconditionally, the model retrieves candidate error windows by context similarity, separates each error's source provenance, and fits a residual-source ridge layer with explicit fallback gating and trust calibration.

- `src/core.py`: context matching, error-source decomposition, and forecast evaluation utilities.
- `src/residual_source_ridge.py`: conditional residual-source ridge estimation with fallback and trust calibration.
- `src/paths.py`: lightweight path helpers for local runs.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` runs the residual-source ridge adapter on synthetic multi-regime series and compares unconditional reuse against trust-calibrated transfer.

## Scope

This is the public CIFEr core MVP on synthetic series. The full multi-market volatility and macro benchmark suite is available upon request.
