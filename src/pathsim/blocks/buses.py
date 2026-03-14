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

import numpy as np

from ._block import Block
from ..utils.register import Register


# MODULE LOGGER =========================================================================

_log = logging.getLogger('pathsim.blocks.buses')


# PUBLIC API =============================================================================

__all__ = [
    'BusCreator', 'BusSelector', 'BusMerge', 'BusFunction',
    # Role markers — consumed by Simulation._build_bus_layout()
    '_BusProducer', '_BusMerger', '_BusConsumer',
]


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


# MODULE-LEVEL HELPERS ==================================================================

def _bus_leaf_count(bus):
    """Return the total number of leaf (scalar) signals in a Bus schema."""
    from ..bus import Bus
    n = 0
    for elem in bus.elements:
        if isinstance(elem.data_type, Bus):
            n += _bus_leaf_count(elem.data_type)
        else:
            n += 1
    return n


def _bus_key_index_map(bus, prefix='', start=0):
    """Build ``{dot_path: flat_int_index}`` for every leaf in *bus*.

    Returns ``(map, next_idx)`` where *next_idx* is the next unused flat index.
    Traversal order is depth-first, matching BusCreator key layout.
    """
    from ..bus import Bus
    m, idx = {}, start
    for elem in bus.elements:
        key = f"{prefix}{elem.name}" if prefix else elem.name
        if isinstance(elem.data_type, Bus):
            sub, idx = _bus_key_index_map(elem.data_type, key + '.', idx)
            m.update(sub)
        else:
            m[key] = idx
            idx += 1
    return m, idx


def _bus_key_layout(bus, prefix='', start=0):
    """Build ``[(key, start_idx, size), ...]`` for packing inputs into a flat buffer.

    For leaf elements, ``size=1`` and ``start_idx`` is the flat index.
    For nested Bus elements, ``size`` is the number of leaves and the caller is expected
    to have a nested ndarray for that key.

    Returns ``(layout_list, next_idx)``.
    """
    from ..bus import Bus
    layout = []
    idx = start
    for elem in bus.elements:
        key = (f"{prefix}{elem.name}" if prefix else elem.name).split('.')[-1]
        if isinstance(elem.data_type, Bus):
            sub_count = _bus_leaf_count(elem.data_type)
            layout.append((elem.name, idx, sub_count))
            idx += sub_count
        else:
            layout.append((elem.name, idx, 1))
            idx += 1
    return layout, idx


# ROLE MARKER MIXINS ====================================================================
# Simulation._build_bus_layout() uses these to identify block roles without
# importing concrete class names.  If a block is renamed or a new bus block is
# added, only buses.py needs to change — simulation.py stays stable.

class _BusProducer:
    """Mixin: block that generates a flat float64 ndarray bus output (has _out_index_map).
    BusCreator and BusFunction inherit this (pass-1 producers — no upstream bus deps)."""

class _BusMerger:
    """Mixin: block that merges upstream buses into a new bus (has _out_index_map +
    _copy_plan).  Processed in pass 2 because it needs upstream _out_index_maps first."""

class _BusConsumer:
    """Mixin: block that reads from a flat ndarray bus input (has _flat_indices or
    _in_flat_indices).  BusSelector and BusFunction inherit this (pass-3 consumers)."""


# MISO BLOCKS ===========================================================================

class BusCreator(_BusProducer, Block):
    """Combines multiple input signals into a structured bus (flat float64 ndarray) output.

    Takes N named scalar inputs and packs them into a pre-allocated float64 buffer
    that is passed downstream as a single signal.  Nested buses are supported by
    connecting the output of one BusCreator to a named input port of another.

    At build time (``Simulation._build_bus_layout``), downstream blocks receive
    pre-computed flat indices so that every FPI read is a single numpy fancy-index
    operation rather than a dict traversal.

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

        # Pre-allocated flat float64 output buffer (build-time, always present)
        self._n = _bus_leaf_count(self.bus)
        self._buf = np.zeros(self._n)
        # Key layout: list of (input_key, flat_start, size)
        self._key_layout, _ = _bus_key_layout(self.bus)

        # Build-time index map for downstream injection (set during _build_bus_layout)
        self._out_index_map = None


    def __repr__(self):
        bus_name = self.bus.name
        if bus_name:
            return f"BusCreator(bus={bus_name!r}, keys={self.keys})"
        return f"BusCreator(keys={self.keys})"

    def reset(self):
        """Reset registers and clear any stale bus output."""
        super().reset()

    def update(self, t=None):
        for key, start, size in self._key_layout:
            val = self.inputs[key]
            if size == 1:
                try:
                    self._buf[start] = float(val)
                except (TypeError, ValueError):
                    self._buf[start] = 0.0
            else:
                # Nested bus: val should be a flat ndarray from another BusCreator
                if isinstance(val, np.ndarray) and len(val) >= size:
                    self._buf[start:start + size] = val[:size]
                # else: leave zeros in buffer (nested bus not yet ready or disconnected)
        # Re-store buffer reference if register was reset (register.reset() clears to 0)
        if self.outputs['bus'] is not self._buf:
            self.outputs['bus'] = self._buf


class BusSelector(_BusConsumer, Block):
    """Selects one or more signals from a bus (flat float64 ndarray) and outputs them.

    Takes a single bus input and extracts the requested keys as separate scalar outputs.
    Dot-notation is supported for nested buses (e.g. ``'Sensors.Temperature'``).

    At build time (``Simulation._build_bus_layout``), ``_flat_indices`` is injected so
    that each FPI read is a single numpy fancy-index operation.  Without a simulation
    (unit tests), the block falls back to the dict-traversal path.

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
        # Pre-split dot paths once at init — avoids repeated str.split() in update()
        self._paths  = [tuple(k.split('.')) for k in self.keys]
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        # Float64 outputs: BusSelector extracts individual scalar signals, not buses.
        # Pre-size to len(keys) so the fast path writes all in one slice.
        self.outputs = Register(mapping=self.output_port_labels, size=len(keys))
        # Keys that have already triggered a missing-key warning (warn once per key)
        self._warned_missing = set()
        # Logger — replaced by Simulation with sim.logger so PathView sees it
        self._logger = _log
        # Injected by Simulation._build_bus_layout() — None means "not yet compiled"
        self._flat_indices = None


    def __repr__(self):
        return f"BusSelector(keys={self.keys})"

    def reset(self):
        """Reset registers and clear per-run warning state."""
        super().reset()
        self._warned_missing.clear()

    def update(self, t=None):
        buf = self.inputs['bus']

        if isinstance(buf, np.ndarray):
            if self._flat_indices is not None:
                # Fast path: single numpy fancy-index operation
                self.outputs._data[:len(self.keys)] = buf[self._flat_indices]
                return
            # ndarray but no indices injected yet — no-op (should not happen in sim)
            return

        # Not an ndarray: dict fallback (unit tests) or FPI zero / wrong type
        if not isinstance(buf, dict):
            if not (isinstance(buf, (int, float)) and buf == 0):
                self._logger.warning(
                    "BusSelector received a non-dict input of type %r. "
                    "Expected a bus dict from a BusCreator. "
                    "Check that the connected output port carries a bus signal.",
                    type(buf).__name__,
                )
            return

        # Dict fallback path — unit tests without sim.run()
        for key, path in zip(self.keys, self._paths):
            val = buf
            missing = False
            for part in path:
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
                    key, list(buf.keys()),
                )
                self._warned_missing.add(key)
            self.outputs[key] = val


class BusMerge(_BusMerger, Block):
    """Merges N bus inputs into a single bus output.

    Takes *n* bus inputs and combines their signals into one output bus.
    At build time (``Simulation._build_bus_layout``), a copy plan is injected
    so that each FPI merge is a set of pre-sized numpy slice copies.
    Without a simulation (unit tests), falls back to dict-merge logic.

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
        # Injected by Simulation._build_bus_layout() — None means "not yet compiled"
        self._copy_plan = None   # list of (dst_start, dst_end, src_input_key)
        self._buf = None         # pre-allocated flat float64 merge buffer
        # Build-time output index map for downstream injection
        self._out_index_map = None


    def __repr__(self):
        return f"BusMerge(n={self.n}, on_conflict={self.on_conflict!r})"


    def reset(self):
        """Reset registers and clear per-run conflict warning state."""
        super().reset()
        self._warned_conflicts.clear()


    def update(self, t=None):
        if self._copy_plan is not None:
            # Fast path: copy slices from input flat ndarrays into merged buffer
            for dst_start, dst_end, src_key in self._copy_plan:
                src = self.inputs[src_key]
                if isinstance(src, np.ndarray):
                    size = dst_end - dst_start
                    self._buf[dst_start:dst_end] = src[:size]
                # else: input not yet ready (FPI zero) — leave zeros in buffer
            if self.outputs['bus'] is not self._buf:
                self.outputs['bus'] = self._buf
            return

        # Dict fallback path — unit tests without sim.run()
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

class BusFunction(_BusProducer, _BusConsumer, Block):
    """Applies a callable to selected signals in a bus and produces a new bus output.

    Takes a single bus input, extracts the signals named by *in_keys*
    (dot-notation supported for nested buses), calls *func* with those values
    as positional arguments, and packs the return values into an output bus keyed
    by *out_keys*.

    At build time (``Simulation._build_bus_layout``), ``_in_flat_indices`` is injected
    so that extraction is a single numpy fancy-index operation.  Without a simulation
    (unit tests), falls back to dict traversal.

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
        self.func      = func
        self.in_keys   = list(in_keys)
        self.out_keys  = list(out_keys)
        # Pre-split dot paths once at init — avoids repeated str.split() in update()
        self._in_paths = [tuple(k.split('.')) for k in self.in_keys]
        self.input_port_labels  = {"bus": 0}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        self._warned_missing = set()
        # Logger — replaced by Simulation with sim.logger so PathView sees it
        self._logger = _log
        # Injected by Simulation._build_bus_layout() — None means "not yet compiled"
        self._in_flat_indices = None   # np.ndarray of np.intp
        self._out_buf = None           # pre-allocated float64 output buffer
        # Build-time output index map for downstream injection
        self._out_index_map = None


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
        buf = self.inputs['bus']

        if isinstance(buf, np.ndarray):
            if self._in_flat_indices is not None:
                # Fast path: extract inputs via fancy-index, apply func, write to _out_buf
                vals = buf[self._in_flat_indices]
                result = self.func(*vals)
                if len(self.out_keys) == 1:
                    self._out_buf[0] = result
                else:
                    self._out_buf[:] = result
                if self.outputs['bus'] is not self._out_buf:
                    self.outputs['bus'] = self._out_buf
                return
            # ndarray but no indices — no-op
            return

        # Not an ndarray: dict fallback (unit tests) or FPI zero / wrong type
        if not isinstance(buf, dict):
            if not (isinstance(buf, (int, float)) and buf == 0):
                self._logger.warning(
                    "BusFunction received a non-dict input of type %r. "
                    "Expected a bus dict from a BusCreator. "
                    "Check that the connected output port carries a bus signal.",
                    type(buf).__name__,
                )
            return

        # Dict fallback path — unit tests without sim.run()
        vals = []
        for key, path in zip(self.in_keys, self._in_paths):
            val = buf
            missing = False
            for part in path:
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
                    key, list(buf.keys()),
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
