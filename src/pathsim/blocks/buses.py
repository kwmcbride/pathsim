#########################################################################################
##
##                        BUS CREATOR, SELECTOR, AND MERGE BLOCKS
##                                   (blocks/buses.py)
##
##                                   Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import re
import logging

from ._block import Block
from ..utils.register import Register


# MODULE LOGGER =========================================================================

_log = logging.getLogger('pathsim.blocks.buses')


# PUBLIC API =============================================================================

__all__ = ['BusCreator', 'BusSelector', 'BusMerge', 'BusFunction']


# VALID CONFLICT MODES =================================================================

_CONFLICT_MODES = frozenset({'warn', 'error', 'first', 'last'})

_BAD_KEY = re.compile(r'^\.|\.\.|\.$|^$')


def _validate_dot_keys(keys, block_name):
    """Raise ValueError for empty, leading-dot, trailing-dot, or double-dot keys."""
    for k in keys:
        if _BAD_KEY.search(k):
            raise ValueError(
                f"{block_name}: invalid key {k!r} — keys must be non-empty and "
                f"must not start/end with a dot or contain consecutive dots."
            )


# MISO BLOCKS ===========================================================================

class BusCreator(Block):
    """Combines multiple input signals into a structured bus (dict) output.

    Takes N named scalar inputs and packs them into a single Python dict
    that is passed downstream as one signal.  The dict keys are the signal
    names; nested buses are supported by connecting the output of one
    BusCreator to a named input port of another.

    Parameters
    ----------
    bus : Bus | list[str | BusElement]
        Either a ``Bus`` object that defines the signal names and schema
        metadata, or a plain list of string / ``BusElement`` names (which
        is auto-wrapped into an anonymous ``Bus`` internally).
    """

    def __init__(self, bus, **kwargs):
        from ..bus import Bus, BusElement
        if isinstance(bus, Bus):
            self.bus = bus
        elif isinstance(bus, list):
            self.bus = Bus('', elements=[
                BusElement(k if isinstance(k, str) else k.name) for k in bus
            ])
        else:
            raise TypeError(
                f"BusCreator: 'bus' must be a Bus object or a list of string keys, "
                f"got {type(bus).__name__!r}."
            )
        self.keys = [e.name for e in self.bus.elements]
        if len(self.keys) == 0:
            raise ValueError("BusCreator: bus must have at least one element.")
        seen = set()
        for k in self.keys:
            if k in seen:
                raise ValueError(
                    f"BusCreator: duplicate key '{k}'. All input signal names must be unique."
                )
            seen.add(k)
        self.input_port_labels = {k: i for i, k in enumerate(self.keys)}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs  = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)


    def __repr__(self):
        bus_name = self.bus.name
        if bus_name:
            return f"BusCreator(bus={bus_name!r}, keys={self.keys})"
        return f"BusCreator(keys={self.keys})"

    def reset(self):
        """Reset registers and clear any stale bus output."""
        super().reset()

    def update(self, t=None):
        bus = self.outputs['bus']
        if isinstance(bus, dict):
            # Update in-place to avoid dict allocation on every RK stage.
            for k in self.keys:
                bus[k] = self.inputs[k]
        else:
            # First call: allocate the dict once.
            self.outputs['bus'] = {k: self.inputs[k] for k in self.keys}


class BusSelector(Block):
    """Selects one or more signals from a bus dict and outputs them individually.

    Takes a single bus (dict) input and extracts the requested keys as
    separate scalar outputs.  Dot-notation is supported for nested buses
    (e.g. ``'Sensors.Temperature'``).

    Parameters
    ----------
    keys : list[str]
        Signal names to extract from the bus.
    """

    def __init__(self, keys):
        if len(keys) == 0:
            raise ValueError("BusSelector: 'keys' must not be empty.")
        _validate_dot_keys(keys, 'BusSelector')
        seen = set()
        for k in keys:
            if k in seen:
                raise ValueError(
                    f"BusSelector: duplicate key '{k}'. All output signal names must be unique."
                )
            seen.add(k)
        self.input_port_labels  = {"bus": 0}
        self.output_port_labels = {k: i for i, k in enumerate(keys)}
        super().__init__()
        self.keys    = list(keys)
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        # Keys that have already triggered a missing-key warning (warn once per key)
        self._warned_missing = set()
        # Logger — replaced by Simulation with sim.logger so PathView sees it
        self._logger = _log


    def __repr__(self):
        return f"BusSelector(keys={self.keys})"

    def reset(self):
        """Reset registers and clear per-run warning state."""
        super().reset()
        self._warned_missing.clear()

    def update(self, t=None):
        bus = self.inputs['bus']
        if not isinstance(bus, dict):
            # Register initialises object-dtype entries to 0; that is the expected
            # transient value during the first FPI iteration before the upstream
            # BusCreator has run.  Anything else is a likely mis-connection.
            if not (isinstance(bus, (int, float)) and bus == 0):
                self._logger.warning(
                    "BusSelector received a non-dict input of type %r. "
                    "Expected a bus dict from a BusCreator. "
                    "Check that the connected output port carries a bus signal.",
                    type(bus).__name__,
                )
            return
        for key in self.keys:
            val = bus
            missing = False
            for part in key.split('.'):
                if isinstance(val, dict):
                    if part not in val:
                        missing = True
                        val = 0.0
                        break
                    val = val[part]
                else:
                    missing = True
                    val = 0.0
                    break
            if missing and key not in self._warned_missing:
                self._logger.info(
                    "BUS WARNING: BusSelector key %r not found in bus. "
                    "Available keys: %s. Output will be 0.0.",
                    key, list(bus.keys()),
                )
                self._warned_missing.add(key)
            self.outputs[key] = val


class BusMerge(Block):
    """Merges N bus inputs into a single bus output.

    Takes *n* bus (dict) inputs and combines their key-value pairs into one
    output bus dict.  When the same key appears in more than one input bus,
    the *on_conflict* policy determines the outcome.

    Parameters
    ----------
    n : int
        Number of bus inputs to merge (default: 2).
    on_conflict : str
        How to handle duplicate keys across input buses:

        ``'warn'``  — emit a :class:`UserWarning` once per conflicting key;
        the **last** bus (highest index) wins.  *(default)*

        ``'error'`` — raise :class:`ValueError` on the first conflicting key.

        ``'first'`` — silently keep the value from the **first** bus that
        provides the key.

        ``'last'``  — silently overwrite with the value from the **last** bus.

    Examples
    --------
    Merge two buses::

        merger = BusMerge(n=2)
        Connection(plant_bus[0],  merger['bus_0'])
        Connection(sensor_bus[0], merger['bus_1'])
        Connection(merger[0],     scope[0])
    """

    def __init__(self, n=2, on_conflict='warn', **kwargs):
        if n < 2:
            raise ValueError(f"BusMerge requires at least 2 inputs, got {n}.")
        if on_conflict not in _CONFLICT_MODES:
            raise ValueError(
                f"BusMerge: invalid on_conflict={on_conflict!r}. "
                f"Choose from {sorted(_CONFLICT_MODES)}."
            )
        self.n = n
        self.on_conflict = on_conflict
        self.input_port_labels  = {f'bus_{i}': i for i in range(n)}
        self.output_port_labels = {'bus': 0}
        super().__init__()
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        # Keys that have already triggered a conflict warning (warn once per key)
        self._warned_conflicts = set()
        # Logger — replaced by Simulation with sim.logger so PathView sees it
        self._logger = _log


    def __repr__(self):
        return f"BusMerge(n={self.n}, on_conflict={self.on_conflict!r})"


    def reset(self):
        """Reset registers and clear per-run conflict warning state."""
        super().reset()
        self._warned_conflicts.clear()


    def update(self, t=None):
        bus_out = self.outputs['bus']
        if not isinstance(bus_out, dict):
            bus_out = {}
            self.outputs['bus'] = bus_out
        else:
            bus_out.clear()

        _src = {}   # key -> first source bus index (for conflict reporting)
        for i in range(self.n):
            bus_in = self.inputs[f'bus_{i}']
            if not isinstance(bus_in, dict):
                # FPI transient (value is 0) or wrong connection — skip silently.
                # BusSelector already warns for wrong types; no need to duplicate.
                continue
            for key, val in bus_in.items():
                if key in bus_out:
                    if self.on_conflict == 'error':
                        raise ValueError(
                            f"BusMerge: key conflict — '{key}' appears in both "
                            f"bus_{_src.get(key, '?')} and bus_{i}."
                        )
                    elif self.on_conflict == 'first':
                        continue   # keep existing value
                    elif self.on_conflict == 'warn':
                        if key not in self._warned_conflicts:
                            self._logger.warning(
                                "BusMerge: key %r appears in multiple input buses. "
                                "bus_%s value will be used (last wins).",
                                key, i,
                            )
                            self._warned_conflicts.add(key)
                        # fall through to overwrite (last wins)
                else:
                    _src[key] = i
                bus_out[key] = val


# SISO TRANSFORM BLOCK ====================================================================

class BusFunction(Block):
    """Applies a callable to selected signals in a bus and produces a new bus output.

    Takes a single bus (dict) input, extracts the signals named by
    *in_keys* (dot-notation supported for nested buses), calls *func*
    with those values as positional arguments, and packs the return
    values into a new bus dict keyed by *out_keys*.

    This avoids the ``BusSelector → processing_block(s) → BusCreator``
    boilerplate for simple per-signal transformations.

    Parameters
    ----------
    func : callable
        Function that receives ``len(in_keys)`` positional arguments
        (the extracted signal values in order) and returns either a
        single value (when ``len(out_keys) == 1``) or a sequence whose
        length matches *out_keys*.
    in_keys : list[str]
        Keys to extract from the input bus.  Dot-notation is supported
        for nested buses (e.g. ``'Sensors.Temperature'``).
    out_keys : list[str]
        Key names for the output bus.  Must match the number of values
        returned by *func*.

    Examples
    --------
    Convert temperature from Celsius to Fahrenheit::

        converter = BusFunction(
            func=lambda T: T * 1.8 + 32,
            in_keys=['Temperature'],
            out_keys=['Temperature_F'],
        )
        Connection(sensor_bus[0], converter['bus'])
        Connection(converter[0],  display['bus'])

    Scale two signals jointly::

        scaler = BusFunction(
            func=lambda T, H: (T * 1.8 + 32, H / 100.0),
            in_keys=['Temperature', 'Humidity'],
            out_keys=['Temperature_F', 'Humidity_fraction'],
        )
    """

    def __init__(self, func, in_keys, out_keys, **kwargs):
        if not callable(func):
            raise TypeError(
                f"BusFunction: 'func' must be callable, got {type(func).__name__!r}."
            )
        if len(in_keys) == 0:
            raise ValueError("BusFunction: 'in_keys' must not be empty.")
        if len(out_keys) == 0:
            raise ValueError("BusFunction: 'out_keys' must not be empty.")
        _validate_dot_keys(in_keys, 'BusFunction in_keys')
        _validate_dot_keys(out_keys, 'BusFunction out_keys')
        for label, keys in [('in_keys', in_keys), ('out_keys', out_keys)]:
            seen = set()
            for k in keys:
                if k in seen:
                    raise ValueError(
                        f"BusFunction: duplicate key '{k}' in {label}."
                    )
                seen.add(k)
        self.func     = func
        self.in_keys  = list(in_keys)
        self.out_keys = list(out_keys)
        self.input_port_labels  = {"bus": 0}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        self._warned_missing = set()
        # Logger — replaced by Simulation with sim.logger so PathView sees it
        self._logger = _log


    def __repr__(self):
        fn_name = getattr(self.func, '__name__', repr(self.func))
        return (
            f"BusFunction(func={fn_name!r}, "
            f"in_keys={self.in_keys}, out_keys={self.out_keys})"
        )

    def __deepcopy__(self, memo):
        import copy
        # deepcopy does not rebind lambda closures, so construct a fresh instance
        # (same callable reference is correct — the func itself is not copied).
        new = BusFunction(self.func, list(self.in_keys), list(self.out_keys))
        new.inputs  = copy.deepcopy(self.inputs,  memo)
        new.outputs = copy.deepcopy(self.outputs, memo)
        new._warned_missing = set()
        new._logger = self._logger
        memo[id(self)] = new
        return new


    def reset(self):
        """Reset registers and clear per-run warning state."""
        super().reset()
        self._warned_missing.clear()


    def update(self, t=None):
        bus = self.inputs['bus']
        if not isinstance(bus, dict):
            if not (isinstance(bus, (int, float)) and bus == 0):
                self._logger.warning(
                    "BusFunction received a non-dict input of type %r. "
                    "Expected a bus dict from a BusCreator. "
                    "Check that the connected output port carries a bus signal.",
                    type(bus).__name__,
                )
            return

        # Extract input values (dot-notation supported).
        vals = []
        for key in self.in_keys:
            val = bus
            missing = False
            for part in key.split('.'):
                if isinstance(val, dict):
                    if part not in val:
                        missing = True
                        val = 0.0
                        break
                    val = val[part]
                else:
                    missing = True
                    val = 0.0
                    break
            if missing and key not in self._warned_missing:
                self._logger.info(
                    "BUS WARNING: BusFunction key %r not found in bus. "
                    "Available keys: %s. Output will be 0.0.",
                    key, list(bus.keys()),
                )
                self._warned_missing.add(key)
            vals.append(val)

        # Call the user function with the extracted values.
        result = self.func(*vals)

        # Pack results into the output bus dict.
        if len(self.out_keys) == 1:
            out_vals = (result,)
        else:
            out_vals = tuple(result)

        bus_out = self.outputs['bus']
        if isinstance(bus_out, dict):
            for k, v in zip(self.out_keys, out_vals):
                bus_out[k] = v
        else:
            self.outputs['bus'] = dict(zip(self.out_keys, out_vals))
