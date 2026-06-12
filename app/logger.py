import logging
import os
import sys

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE")

# Configure logger
logger = logging.getLogger("acela")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Clear existing handlers to prevent duplicates
if logger.handlers:
    logger.handlers.clear()

# Create formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional File handler (activated if LOG_FILE environment variable is set)
if LOG_FILE:
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging also redirected to file: {LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to setup file logging to {LOG_FILE}: {e}")
