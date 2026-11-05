# Changelog

## 0.1.0 - MVP v0

- Added PyG EllipticBitcoinDataset loader with explicit time-step recovery.
- Added chronological validation and train/test split (steps 1-29 / 30-34 / 1-34 / 35-49).
- Added per-time-step NetworkX structural features and train-only standardization.
- Added Random Forest and XGBoost raw/engineered baselines.
- Added Node2Vec + Logistic Regression baseline.
- Added two-layer GraphSAGE baseline and validation-based fallback signal.
- Added PR-AUC, illicit-class F1, ROC-AUC, and horizon reports.
- Added synthetic unit tests and GitHub Actions CI.
