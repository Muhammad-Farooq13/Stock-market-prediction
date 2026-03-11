"""
Logging Utilities Module
Provides centralized logging configuration
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """Custom logger class for the project"""

    def __init__(self, name: str, log_dir: str = None, level: str = "INFO"):
        """
        Initialize logger

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.name = name
        self.logger = logging.getLogger(name)

        # Set level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log_dir provided)
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get logger instance"""
        return self.logger

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


def setup_logger(name: str, log_dir: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Setup and return a logger

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger
    """
    logger_instance = Logger(name, log_dir, level)
    return logger_instance.get_logger()


def main():
    """Main function to demonstrate logging"""
    # Setup logger
    logger = setup_logger("demo", log_dir="logs", level="DEBUG")

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n=== Logging Demo Complete ===")


if __name__ == "__main__":
    main()
