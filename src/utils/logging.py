"""
logging.py

Project-wide logging utilities.
"""

from pathlib import Path
import logging


def get_logger(
    name: str,
    log_directory: str | Path = "logs",
    log_file: str = "project.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    log_directory : str | Path
        Directory where log files are written.

    log_file : str
        Name of the log file.

    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Log file output
    file_handler = logging.FileHandler(
        log_directory / log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger