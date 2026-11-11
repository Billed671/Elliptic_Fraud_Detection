"""Command-line entry point."""

from __future__ import annotations

import argparse

from elliptic_fraud.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elliptic-fraud",
        description="Temporal illicit-transaction classification on the Elliptic Bitcoin graph.",
    )
    parser.add_argument("--data-dir", default="data", help="PyG dataset root (default: data).")
    parser.add_argument("--output-dir", default="outputs", help="Artifact directory (default: outputs).")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated subset of: classical,graphsage,node2vec (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graphsage-epochs", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    valid = {"classical", "graphsage", "node2vec"}
    requested = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    if args.models.strip().lower() == "all":
        requested = ("classical", "graphsage", "node2vec")
    unknown = set(requested) - valid
    if unknown:
        raise SystemExit(f"Unknown model group(s): {', '.join(sorted(unknown))}. Choose from {sorted(valid)}.")
    if not requested:
        raise SystemExit("Choose at least one model group.")
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=requested,
        seed=args.seed,
        graphsage_epochs=args.graphsage_epochs,
    )


if __name__ == "__main__":
    main()
