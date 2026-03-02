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


@pytest.fixture
def pathsim_logs():
    """Capture INFO-level (and above) log records from the 'pathsim' logger hierarchy.

    Temporarily lowers the 'pathsim' logger to INFO so that bus-block advisory
    messages (emitted at INFO level so PathView surfaces them) are captured.

    Records have a pre-computed ``.message`` attribute set eagerly in ``emit()``.
    """
    class _CapHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            if record.levelno >= logging.INFO:
                record.message = record.getMessage()
                self.records.append(record)

    h = _CapHandler()
    pathsim_logger = logging.getLogger('pathsim')
    old_level = pathsim_logger.level
    pathsim_logger.setLevel(logging.INFO)
    pathsim_logger.addHandler(h)
    yield h.records
    pathsim_logger.removeHandler(h)
    pathsim_logger.setLevel(old_level)
