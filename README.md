# History Transfer (CIFEr core)

Conditional historical-error retrieval, cross-market ridge source transfer, and QLIKE calibration for multi-market realized-volatility forecasting (IEEE SSCI 2027 / CIFEr, accepted full paper).

Cross-market error reuse mixes two separate effects: finding similar historical dates using joint market states versus transferring other markets' residual errors into a target market.

- `src/residual_source_ridge.py`: QLIKE loss (`ql`), equal-market macro scoring (`macro`), cross-market ridge weights excluding own-market leakage, and validation-year multiplier calibration (`calibrate`).
- `src/core.py`: original CIFEr PyTorch retrieval encoder (`LearnedGlobalEncoder`) and market graph (`RegimeMarketGraph`) kept as reference.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` fits zero-diagonal cross-market ridge weights on a synthetic 6-market error panel, calibrates `gamma` on a validation split, and prints held-out macro QLIKE before and after source transfer.

## Scope

This is the CIFEr original codebase (not the separate FMTS revision). Reproducing the 31-index empirical tables requires the Oxford-Man realized-volatility data and PyTorch.
