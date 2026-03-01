"""Tests for structured logging configuration."""

import logging
import sys

import pytest
import structlog

from vaanidub.logging_config import setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    """Save and restore logging state to avoid polluting other tests."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


class TestSetupLogging:
    def test_setup_logging_console(self):
        """Console format should configure a structlog ProcessorFormatter."""
        setup_logging(log_level="INFO", log_format="console")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        assert isinstance(formatter, structlog.stdlib.ProcessorFormatter)

    def test_setup_logging_json(self):
        """JSON format should configure JSONRenderer."""
        setup_logging(log_level="INFO", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        formatter = handler.formatter
        assert formatter is not None

    def test_setup_logging_level_debug(self):
        setup_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_level_warning(self):
        setup_logging(log_level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_setup_logging_level_info(self):
        setup_logging(log_level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_noisy_libraries_quieted(self):
        setup_logging()
        for lib_name in ["urllib3", "httpx", "httpcore", "transformers"]:
            lib_logger = logging.getLogger(lib_name)
            assert lib_logger.level >= logging.WARNING

    def test_previous_handlers_cleared(self):
        """Old handlers should be removed, leaving exactly 1."""
        root = logging.getLogger()
        # Add a dummy handler
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        setup_logging()
        assert len(root.handlers) == 1

    def test_handler_writes_to_stdout(self):
        setup_logging()
        root = logging.getLogger()
        handler = root.handlers[0]
        assert handler.stream is sys.stdout
