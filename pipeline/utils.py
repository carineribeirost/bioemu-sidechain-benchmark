import logging
import sys


def setup_logger(name: str = "ensemble_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


log = setup_logger()
