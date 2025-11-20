import logging

import pytest

import gwrefpy as gr

logger = logging.getLogger(__name__)


def test_set_log_level():
    gr.set_log_level("DEBUG")
    # Get the StreamHandler
    streamhandler = next(
        h for h in logger.parent.handlers if isinstance(h, logging.StreamHandler)
    )
    # Check that the handler level is set to DEBUG
    assert streamhandler.level == logging.DEBUG
    gr.set_log_level("INFO")
    assert streamhandler.level == logging.INFO
    gr.set_log_level("WARNING")
    assert streamhandler.level == logging.WARNING
    gr.set_log_level("ERROR")
    assert streamhandler.level == logging.ERROR
    gr.set_log_level("CRITICAL")
    assert streamhandler.level == logging.CRITICAL


def test_set_log_level_invalid():
    with pytest.raises(ValueError, match="Invalid log level: INVALID_LEVEL"):
        gr.set_log_level("INVALID_LEVEL")


def test_enable_file_logging(tmp_path):
    log_file = tmp_path / "test_log.log"
    gr.enable_file_logging(str(log_file))
    # Get the FileHandler
    filehandler = next(
        h for h in logger.parent.handlers if isinstance(h, logging.FileHandler)
    )
    assert filehandler is not None
    # Log a test message
    test_message = "This is a test log message."
    logger.info(test_message)
    # Flush the handler to ensure the message is written
    filehandler.flush()
    # Check that the message was written to the file
    with open(log_file) as f:
        log_contents = f.read()
        assert test_message in log_contents
