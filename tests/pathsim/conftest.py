import logging
import pytest


@pytest.fixture
def pathsim_warnings():
    """Capture WARNING-level log records emitted by the 'pathsim' logger hierarchy.

    Records have a pre-computed ``.message`` string attribute set eagerly in
    ``emit()``, so tests can do ``r.message`` without first calling a formatter.
    """
    class _CapHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            if record.levelno >= logging.WARNING:
                record.message = record.getMessage()
                self.records.append(record)

    h = _CapHandler()
    pathsim_logger = logging.getLogger('pathsim')
    pathsim_logger.addHandler(h)
    yield h.records
    pathsim_logger.removeHandler(h)
