# Elliptic Bitcoin Fraud Detection — MVP v0

**A temporal, imbalanced node-classification benchmark for illicit Bitcoin transactions.**

This repository implements the A4 proposal from Rustem Gilmetdinov, Zamir Safin, and Denis Beliaev. It compares tabular machine learning, graph embeddings, and a two-layer GraphSAGE baseline on the PyTorch Geometric (PyG) `EllipticBitcoinDataset`.

> **Primary metric: PR-AUC.** Accuracy is intentionally not reported. The illicit class is the positive class (`1`); licit is `0`; unknown is `2` and is excluded from supervised fitting and metric computation.

## MVP scope

| Component | MVP v0 implementation |
|---|---|
| Dataset | PyG `EllipticBitcoinDataset` |
| Graph | Directed transaction-flow edges |
| Numeric node features | 165 original features from `data.x` |
| Time feature | Raw `time_step` column appended to tabular features (166 columns before graph metrics) |
| Structural features | PageRank, degree centrality, clustering coefficient (NetworkX) |
| Classical baselines | Random Forest and XGBoost, each with raw+time and engineered feature variants |
| Embedding baseline | Node2Vec embeddings + Logistic Regression |
| GNN baseline | Two-layer GraphSAGE using original node features and graph structure |
| Primary metric | PR-AUC / average precision |
| Secondary metrics | Illicit-class F1 and ROC-AUC |
| Test horizons | Steps 35–39, 40–44, and 45–49 |

## Dataset facts and label balance

The current PyG documentation describes **203,769 nodes, 234,355 directed edges, 165 numeric node features, and two supervised classes**. The dataset spans 49 time steps. PyG maps labels to `0 = licit`, `1 = illicit`, and `2 = unknown`.

The proposal estimates that about 4% of labeled nodes are illicit. The current PyG documentation lists 4,545 illicit and 42,019 licit nodes, which is about **9.8% illicit among known labels** (and about 2.2% of all nodes). To avoid relying on stale or inconsistent counts, the loader prints the actual counts from the downloaded dataset every run. The MVP does not hard-code class prevalence.

The PyG `Data` object exposes the 165 numeric features but does not expose the time-step column as a `Data` attribute in the current dataset implementation. The loader therefore reads column 1 (`time_step`) from the raw feature CSV and checks that its row count aligns with `data.x` and `data.y`.

## Temporal evaluation protocol

The proposal specifies training on steps **1–34** and testing on **35–49**, with forecast horizons measured from the step-34 cutoff. MVP v0 makes the validation protocol explicit:

| Purpose | Time steps | Use |
|---|---:|---|
| Early training | 1–29 | Fit initial models |
| Validation | 30–34 | Compare models and select the F1 decision threshold |
| Final training | 1–34 | Refit selected/final models |
| Test horizon 1 | 35–39 | 1–5 steps ahead |
| Test horizon 2 | 40–44 | 6–10 steps ahead |
| Test horizon 3 | 45–49 | 11–15 steps ahead |

Only known labels are included in the train, validation, and test masks. Unknown nodes remain in the graph and can contribute to unsupervised topology/message passing, but never enter supervised loss or evaluation metrics.

The validation split is chronological and is used only for model comparison and threshold selection. After validation, models are refit using all known labels from steps 1–34. The F1 threshold is frozen from validation before test scoring. PR-AUC does not require a threshold.

### What this split does and does not claim

This is a **temporal holdout benchmark**, not a fully prospective simulation in which the model at step 34 must forecast unseen node features before those transactions arrive. GraphSAGE receives the complete graph as a transductive input, and Node2Vec is intentionally trained on the observed full graph topology. Neither method sees test labels, but both can see test-node connectivity; Node2Vec is therefore explicitly a transductive topology-only baseline. Use a separate inductive/streaming protocol before making deployment claims.

Structural metrics are computed separately within each time-step-induced subgraph. PageRank is computed on the directed graph; degree centrality and clustering coefficient are computed on its undirected projection. This prevents an edge from a different time step from contributing to a node's structural feature.

## Repository layout

```text
elliptic-fraud-mvp/
├── .github/workflows/tests.yml
├── data/                         # downloaded data is gitignored
├── outputs/                      # generated models/reports are gitignored
├── src/elliptic_fraud/
│   ├── cli.py                     # command-line interface
│   ├── data.py                    # PyG loading + time-step alignment
│   ├── experiment.py               # end-to-end orchestration
│   ├── features.py                 # structural features + train-only scaling
│   ├── metrics.py                  # frozen metrics + horizon evaluation
│   ├── splits.py                   # temporal masks
│   └── models/
│       ├── classical.py            # Random Forest + XGBoost
│       ├── graphsage.py            # two-layer GraphSAGE
│       └── node2vec.py             # Node2Vec + Logistic Regression
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Quick start

Python 3.10+ is required. A virtual environment is recommended.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

PyTorch and PyG installation can depend on your OS, Python, CUDA, and PyTorch versions. If the default installation does not select a compatible build, first install the correct PyTorch build from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Node2Vec additionally needs a compatible PyG random-walk sampling extension (`pyg-lib` or `torch-cluster`); see the PyG guide for the matching wheel.

## Run the experiment

The first run downloads the dataset through PyG into `data/`.

```bash
# Full MVP: Random Forest, XGBoost, Node2Vec, and GraphSAGE
elliptic-fraud --data-dir data --output-dir outputs --models all

# Only tabular baselines (fastest first run after data download)
elliptic-fraud --models classical

# Run one graph baseline
elliptic-fraud --models graphsage
elliptic-fraud --models node2vec
```

Equivalent module invocation:

```bash
python -m elliptic_fraud.cli --models all --seed 42 --graphsage-epochs 20
```

To use fewer GraphSAGE epochs for a smoke run, pass `--graphsage-epochs 3`. The proposal's fallback check uses 20 epochs by default.

## Outputs

After a successful run:

```text
outputs/
├── models/
│   ├── random_forest_raw.joblib
│   ├── random_forest_engineered.joblib
│   ├── xgboost_raw.joblib
│   ├── xgboost_engineered.joblib
│   ├── node2vec_logistic.joblib
│   ├── node2vec_embeddings.npy
│   └── graphsage.pt
└── reports/
    ├── metrics_by_horizon.csv
    └── experiment_summary.json
```

`metrics_by_horizon.csv` contains model, horizon, number of evaluated labeled nodes, number of illicit nodes, PR-AUC, ROC-AUC, and illicit-class F1. `experiment_summary.json` records class counts, the temporal split, validation PR-AUC values, F1 thresholds, and the fallback decision.

No experimental scores are included in this repository before a real run. Do not add fabricated or illustrative scores to the results table.

## Evaluation and fallback rule

The metrics are fixed before training:

1. **PR-AUC (average precision)** — primary ranking metric under class imbalance.
2. **F1 for illicit transactions** — secondary metric; the classification threshold is selected on steps 30–34 only.
3. **ROC-AUC** — secondary ranking metric.

The fallback is triggered when the best GraphSAGE validation PR-AUC observed during the 20-epoch validation run is below the best classical ML validation PR-AUC. The final report records this as `graphsage_fallback_triggered`. A triggered fallback means the classical ML / Node2Vec pipeline is preferred over plain message passing for this MVP; it does not erase the GraphSAGE baseline results.

## Leakage and reproducibility checklist

- [x] Split by time, never by a random node-level split.
- [x] Exclude unknown labels (`y == 2`) from supervised fitting and scoring.
- [x] Fit every `StandardScaler` on the current training indices only.
- [x] Select the F1 threshold on validation data, never on the test set.
- [x] Report PR-AUC by forecast horizon as well as the full 35–49 test window.
- [x] Use a fixed random seed (default `42`).
- [x] Keep raw dataset files and generated artifacts out of Git.
- [ ] Repeat final runs across multiple seeds and report variability (planned after MVP v0).
- [ ] Add an inductive/streaming evaluation where future topology is unavailable (planned).

## Tests

The tests use small synthetic graphs and do not download Elliptic data.

```bash
pytest -q
```

## Known limitations / MVP v1 ideas

- Full-graph Node2Vec requires a compatible PyG sampling extension and can be computationally expensive.
- GraphSAGE and Node2Vec use a transductive graph view; a strict future-only topology protocol is not implemented yet.
- NetworkX structural feature extraction is CPU-based and may be the slowest preprocessing stage.
- The current evaluation uses a single chronological validation window and one seed.
- No model calibration, confidence intervals, threshold-at-fixed-recall analysis, or deployment/streaming inference API is included.
- H2GCN and other heterophily-aware architectures are intentionally out of scope for MVP v0; GraphSAGE is the simple message-passing baseline from the proposal.

## Data and paper references

- PyG dataset documentation: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html>
- PyG dataset implementation: <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/elliptic.py>
- Original paper: M. Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019 Workshop on Anomaly Detection in Finance. <https://arxiv.org/abs/1908.02591>

The dataset is downloaded by PyG and is not included in this repository. Check the original dataset terms before using it beyond research/educational purposes.

## Team

- Rustem Gilmetdinov
- Zamir Safin
- Denis Beliaev

## License

The code in this repository is released under the MIT License. The dataset is not relicensed by this repository.
