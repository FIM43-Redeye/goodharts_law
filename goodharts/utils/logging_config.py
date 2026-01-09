import logging
import sys

def setup_logging(log_file: str = "simulation.log", level: int = logging.INFO):
    """
    Sets up logging to both console and file.
    """
    # Create logger
    logger = logging.getLogger("goodharts")
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_logger(name: str):
    """
    Returns a child logger for specific modules.
    e.g. get_logger("agent.123") -> "goodharts.agent.123"
    """
    return logging.getLogger(f"goodharts.{name}")
