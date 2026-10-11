"""Data downloading, acquisition and streaming loading utilities."""

import gzip
import logging
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import pandas as pd

from ctr_prediction.config import get_project_root

logger = logging.getLogger(__name__)


def download_criteo_sample(
    url: str = "https://maxhalford.github.io/files/datasets/criteo_sample.csv.gz",
    destination_path: Optional[Path | str] = None,
    force: bool = False,
) -> Path:
    """Download the official Criteo 100k sample if not already present.

    Parameters
    ----------
    url : str
        Source download URL.
    destination_path : Path or str, optional
        Where to save the file. Defaults to data/raw/criteo_sample.csv.gz.
    force : bool, default=False
        If True, re-download even if already present.

    Returns
    -------
    Path
        Absolute path to the downloaded dataset.
    """
    if destination_path is None:
        destination_path = get_project_root() / "data" / "raw" / "criteo_sample.csv.gz"
    else:
        destination_path = Path(destination_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() and not force:
        logger.info(f"Dataset already exists at {destination_path}. Skipping download.")
        return destination_path

    logger.info(f"Downloading Criteo sample from {url} to {destination_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response, open(destination_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    logger.info(f"Download complete: {destination_path} ({destination_path.stat().st_size} bytes)")
    return destination_path


def load_criteo_data(
    file_path: Optional[Path | str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load Criteo sample into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Path or str, optional
        Path to the gzipped or plain CSV.
    nrows : int, optional
        Number of rows to load.

    Returns
    -------
    pd.DataFrame
        Loaded Criteo dataset.
    """
    if file_path is None:
        file_path = get_project_root() / "data" / "raw" / "criteo_sample.csv.gz"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Call download_criteo_sample() first."
        )

    logger.info(f"Loading Criteo data from {file_path} (nrows={nrows})...")
    df = pd.read_csv(file_path, compression="gzip" if str(file_path).endswith(".gz") else None, nrows=nrows)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df
