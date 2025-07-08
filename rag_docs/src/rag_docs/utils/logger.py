import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
LOG_FILE = os.getenv("LOG_FILE", "")
LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))


def setup_logger():
    logger = logging.getLogger("app")
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        logger.setLevel(getattr(logging, LOG_LEVEL))
    except AttributeError:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)

    if LOG_FILE:
        try:
            log_path = Path(LOG_FILE).parent
            if str(log_path) != '.':
                log_path.mkdir(exist_ok=True, parents=True)

            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=LOG_MAX_SIZE,
                backupCount=LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Logging to file: {LOG_FILE}")
            return logger
        except (PermissionError, OSError) as e:
            print(f"Error setting up file logging: {str(e)}")
            print("Falling back to console logging")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    print("Logging to console")

    return logger


# Create a single logger instance to be imported throughout the package
logger = setup_logger()
