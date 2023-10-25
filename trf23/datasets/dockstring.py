"""Code for dockstring dataset."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "dockstring_files"
DATASET_FILENAME = "dockstring-dataset.tsv"
DATASET_URL = "https://figshare.com/ndownloader/files/35948138"
SPLIT_FILENAME = "cluster_split.tsv"
SPLIT_URL = "https://figshare.com/ndownloader/files/35948123"


def load_dockstring_dataframes(
    dataset_dir: str, limit_num_train: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dockstring dataset from the specified directory.

    Args:
        dataset_dir: Path to the directory containing the dockstring dataset.

    Returns:
        train_df, test_df: DataFrames containing the training and test data.
    """
    logger.debug("Starting to load datasets...")

    # Ensure file paths are present
    dataset_path = Path(dataset_dir) / DATASET_FILENAME
    assert dataset_path.exists()

    dataset_split_path = Path(dataset_dir) / SPLIT_FILENAME
    assert dataset_split_path.exists()

    # Copied from data loading notebook
    df = pd.read_csv(dataset_path, sep="\t").set_index("inchikey")
    splits = (
        pd.read_csv(dataset_split_path, sep="\t")
        .set_index("inchikey")  # use same index as dataset
        .loc[df.index]  # re-order to match the dataset
    )

    df_train = df[splits["split"] == "train"]
    df_test = df[splits["split"] == "test"]

    # Optionally limit train data size by subsampling without replacement
    if limit_num_train is not None:
        assert limit_num_train <= len(df_train)
        df_train = df_train.sample(n=limit_num_train, replace=False)

    logger.debug("Finished loading datasets.")
    return df_train, df_test


def ensure_dataset_downloaded() -> None:
    """Checks if the dockstring dataset is present, and if not downloads it."""
    DATASET_DIR.mkdir(parents=False, exist_ok=True)  # ensure directory exists

    dataset_path = DATASET_DIR / DATASET_FILENAME
    if not dataset_path.exists():
        print("Dataset not present. Downloading from figshare...")
        urllib.request.urlretrieve(DATASET_URL, filename=dataset_path)

    split_path = DATASET_DIR / SPLIT_FILENAME
    if not split_path.exists():
        print("Dataset split not present. Downloading from figshare...")
        urllib.request.urlretrieve(SPLIT_URL, filename=split_path)


def get_train_test_smiles(target_name: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    ensure_dataset_downloaded()
    df_train, df_test = load_dockstring_dataframes(str(DATASET_DIR))

    # First, remove NaNs
    df_train = df_train[["smiles", target_name]].dropna()
    df_test = df_test[["smiles", target_name]].dropna()

    # Move everything to lists/arrays
    smiles_train = df_train.smiles.to_list()
    smiles_test = df_test.smiles.to_list()
    y_train = df_train[target_name].to_numpy()
    y_test = df_test[target_name].to_numpy()

    # Clip to max of 5.0
    y_train = np.minimum(y_train, 5.0)
    y_test = np.minimum(y_test, 5.0)

    return smiles_train, smiles_test, y_train, y_test
