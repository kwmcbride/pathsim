from importlib import metadata

try:
    __version__ = metadata.version("pathsim")
except Exception:
    __version__ = "unknown"

from .simulation import Simulation
from .connection import Connection, Duplex
from .bus import Bus, BusElement
from .subsystem import Subsystem, Interface
from .utils.logger import LoggerManager
from .bus import Bus, BusElement