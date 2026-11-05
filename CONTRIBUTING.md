# Contributing

1. Create a branch for your change.
2. Keep the chronological split and metric definitions unchanged unless the experiment protocol is explicitly versioned.
3. Never fit preprocessing statistics on validation or test nodes.
4. Do not add raw Elliptic data, generated embeddings, model checkpoints, or result artifacts to Git.
5. Run `pytest -q` before opening a pull request.
6. Include the exact command, seed, environment, and split definition with any reported experiment.
