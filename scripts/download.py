"""
Download benchmark datasets for TabNet Studio.

Datasets:
- Adult Census Income (UCI ID: 2)
- Forest Cover Type (UCI ID: 31)

Usage:
    python scripts/download_datasets.py

    python scripts/download_datasets.py --dataset adult

    python scripts/download_datasets.py --dataset covertype
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATA_DIR = Path("data/raw")


DATASETS = {
    "adult": {
        "id": 2,
        "filename": "adult.csv",
    },
    "covertype": {
        "id": 31,
        "filename": "covertype.csv",
    },
}


def download_dataset(name: str) -> None:
    """Download a dataset from the UCI ML Repository."""

    config = DATASETS[name]

    print(f"\nDownloading {name}...")

    dataset = fetch_ucirepo(id=config["id"])

    X = dataset.data.features
    y = dataset.data.targets

    if isinstance(y, pd.Series):
        y = y.to_frame()

    df = pd.concat([X, y], axis=1)

    save_dir = DATA_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / config["filename"]
    df.to_csv(save_path, index=False)

    print(f"[OK] Saved to: {save_path}")
    print(f"  Shape: {df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for TabNet Studio."
    )

    parser.add_argument(
        "--dataset",
        choices=["adult", "covertype", "all"],
        default="all",
        help="Dataset to download.",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in DATASETS:
            download_dataset(dataset)
    else:
        download_dataset(args.dataset)

    print("\nAll requested datasets downloaded successfully.")


if __name__ == "__main__":
    main()