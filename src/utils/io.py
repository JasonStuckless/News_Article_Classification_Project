"""
io.py

Generic file input/output utilities used throughout the project.
"""

from pathlib import Path
import pickle

import pandas as pd


def ensure_directory(path: str | Path) -> None:
    """
    Create a directory (and any missing parent directories)
    if it does not already exist.

    Parameters
    ----------
    path : str | Path
        Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(file_path)


def write_csv(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    index: bool = False
) -> None:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to save.

    file_path : str | Path
        Destination CSV file.

    index : bool
        Whether to include the DataFrame index.
    """
    file_path = Path(file_path)

    ensure_directory(file_path.parent)

    dataframe.to_csv(file_path, index=index)


def save_pickle(
    obj,
    file_path: str | Path
) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    obj
        Python object to save.

    file_path : str | Path
        Destination pickle file.
    """
    file_path = Path(file_path)

    ensure_directory(file_path.parent)

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path: str | Path):
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    file_path : str | Path
        Pickle file to load.

    Returns
    -------
    object
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)