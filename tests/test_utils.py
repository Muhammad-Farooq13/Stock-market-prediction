"""
Unit tests for utils module
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config, get_config  # noqa: E402
from src.utils.logger import Logger, setup_logger  # noqa: E402


class TestConfig:
    """Test cases for Config class"""

    def test_initialization(self):
        """Test Config initialization"""
        config = Config()

        assert config.PROJECT_ROOT is not None
        assert config.DATA_DIR.exists()
        assert config.MODELS_DIR.exists()
        assert config.LOGS_DIR.exists()

    def test_default_values(self):
        """Test default configuration values"""
        config = Config()

        assert config.RANDOM_STATE == 42
        assert config.FLASK_PORT == 5000
        assert isinstance(config.MA_WINDOWS, list)
        assert isinstance(config.EMA_SPANS, list)

    def test_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "project_root" in config_dict
        assert "data_dir" in config_dict
        assert "models_dir" in config_dict


class TestLogger:
    """Test cases for Logger class"""

    def test_initialization(self):
        """Test Logger initialization"""
        logger = Logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.logger is not None

    def test_log_levels(self):
        """Test different log levels"""
        logger = Logger("test_logger", level="DEBUG")

        # Test that logging doesn't raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_setup_logger(self):
        """Test logger setup function"""
        logger = setup_logger("test_setup_logger", level="INFO")

        assert logger is not None
        assert logger.name == "test_setup_logger"


def test_config_singleton():
    """Test that get_config returns the same instance"""
    config1 = get_config()
    config2 = get_config()

    assert config1 is config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
