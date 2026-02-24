import numpy as np
#########################################################################################
##
##                             BUS CREATOR AND SELECTOR BLOCKS
##                                   (blocks/buses.py)
##
##                                   Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block
from ..utils.register import Register


# HELPERS ===============================================================================

def _fill_flat_layout(elements, prefix, layout, offset):
    """Recursively populate *layout* (dotted-key → flat index) from bus elements.

    Parameters
    ----------
    elements : list[BusElement]
    prefix : str
        Dotted prefix accumulated so far (empty string at the top level).
    layout : dict
        Output mapping being built in-place.
    offset : int
        Current flat-array position.

    Returns
    -------
    offset : int
        Updated position after all elements have been processed.
    """
    for elem in elements:
        key = f"{prefix}.{elem.name}" if prefix else elem.name
        if elem.is_nested():
            offset = _fill_flat_layout(elem.data_type.elements, key, layout, offset)
        else:
            layout[key] = offset
            offset += 1
    return offset


# MISO BLOCKS ===========================================================================

class BusCreator(Block):
    """Combines multiple input signals into a structured bus output.

    Before the simulation loop starts, ``Simulation._prepare_buses()``
    calls :meth:`prepare`, which pre-computes a flat ``float64`` array
    layout so that :meth:`update` never allocates Python objects at
    runtime.  If ``prepare`` has not been called the block falls back to
    building a Python ``dict`` on every step.

    Parameters
    ----------
    keys : list[str | BusElement] | Bus
        Either a list of plain string (or ``BusElement``) key names, or a
        ``Bus`` object that defines the structure including nested buses.
    bus : Bus, optional
        Deprecated positional form — prefer passing a ``Bus`` as *keys*.

    Attributes
    ----------
    flat_layout : dict[str, int] or None
        Dotted-key → flat-index mapping set by :meth:`prepare`.
    """

    def __init__(self, keys=None, bus=None, **kwargs):
        from ..bus import Bus
        if bus is not None:
            self.bus = bus
            self.keys = [e.name for e in bus.elements]
        elif isinstance(keys, Bus):
            # Accepting a Bus object via the keys= parameter is equivalent to bus=
            self.bus = keys
            self.keys = [e.name for e in keys.elements]
        else:
            self.bus = None
            self.keys = [k.name if hasattr(k, 'name') else k for k in keys]
        self.input_port_labels = {k: i for i, k in enumerate(self.keys)}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)

        # Populated by prepare(); None means "not yet prepared"
        self.flat_layout = None
        self._flat_output = None
        self._flat_input_info = None  # list of (int_input_idx, out_start, size)


    def prepare(self):
        """Pre-compute flat array layout and allocate the output buffer.

        Called once by ``Simulation._prepare_buses()`` before the
        simulation loop.  After this call :meth:`update` operates
        entirely on pre-allocated ``float64`` arrays with no Python
        object creation.
        """
        # ------------------------------------------------------------------
        # 1.  Compute the flat layout (dotted-key → flat index)
        # ------------------------------------------------------------------
        layout = {}
        offset = 0
        if self.bus is not None:
            offset = _fill_flat_layout(self.bus.elements, '', layout, offset)
        else:
            for k in self.keys:
                layout[k] = offset
                offset += 1
        self.flat_layout = layout

        # ------------------------------------------------------------------
        # 2.  Pre-allocate the output float64 array
        # ------------------------------------------------------------------
        self._flat_output = np.zeros(offset, dtype=np.float64)

        # ------------------------------------------------------------------
        # 3.  Pre-compute (int_input_idx, out_start, size) for each slot
        # ------------------------------------------------------------------
        self._flat_input_info = []
        slot_offset = 0
        if self.bus is not None:
            for elem in self.bus.elements:
                input_idx = self.input_port_labels[elem.name]
                if elem.is_nested():
                    size = len(elem.data_type.get_leaf_elements())
                else:
                    size = 1
                self._flat_input_info.append((input_idx, slot_offset, size))
                slot_offset += size
        else:
            for k in self.keys:
                self._flat_input_info.append((self.input_port_labels[k], slot_offset, 1))
                slot_offset += 1

        # ------------------------------------------------------------------
        # 4.  Ensure registers are large enough and pre-link the output
        # ------------------------------------------------------------------
        if self._flat_input_info:
            max_in = max(info[0] for info in self._flat_input_info)
            self.inputs.resize(max_in + 1)
        self.outputs.resize(1)
        # Store the reference once; update() mutates it in-place
        self.outputs._data[0] = self._flat_output


    def update(self, t=None):
        if self._flat_output is not None:
            # ----------------------------------------------------------
            # Fast path: fill pre-allocated float64 array in-place
            # ----------------------------------------------------------
            inp = self.inputs._data
            out = self._flat_output
            for input_idx, out_start, size in self._flat_input_info:
                val = inp[input_idx]
                if size == 1:
                    out[out_start] = val
                else:
                    # nested sub-bus: val is the sub-BusCreator's float64 array
                    out[out_start:out_start + size] = val
            # Re-link if reset() replaced the stored reference with 0.0
            if self.outputs._data[0] is not self._flat_output:
                self.outputs._data[0] = self._flat_output
        else:
            # ----------------------------------------------------------
            # Fallback: dict (used before prepare() is called)
            # ----------------------------------------------------------
            self.outputs['bus'] = {k: self.inputs[k] for k in self.keys}


class BusSelector(Block):
    """Selects one or more signals from a bus and outputs them individually.

    Before the simulation loop starts, ``Simulation._prepare_buses()``
    calls :meth:`prepare` with the flat layout produced by the upstream
    :class:`BusCreator`.  After that :meth:`update` uses direct integer
    indexing into the flat ``float64`` array — no dict lookups.

    Parameters
    ----------
    keys : list[str]
        Signal names to extract, supporting dot notation for nested buses
        (e.g. ``'Sensors.Temperature'``).
    """

    def __init__(self, keys, bus_keys=None):
        self.input_port_labels = {"bus": 0}
        self.output_port_labels = {k: i for i, k in enumerate(keys)}
        super().__init__()
        self.keys = list(keys)
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)

        # Populated by prepare(); None means "not yet prepared"
        self.flat_layout = None
        self._sel_indices = None   # flat-array index for each selected key
        self._out_int_idx = None   # integer output-register index for each key


    def prepare(self, flat_layout):
        """Pre-compute flat-array access indices for the selected keys.

        Called by ``Simulation._prepare_buses()`` after the upstream
        :class:`BusCreator` has been prepared.

        Parameters
        ----------
        flat_layout : dict[str, int]
            Dotted-key → flat-index mapping from ``BusCreator.flat_layout``.
        """
        self.flat_layout = flat_layout
        self._sel_indices = [flat_layout.get(k, 0) for k in self.keys]
        self._out_int_idx = [self.output_port_labels[k] for k in self.keys]
        # Ensure the outputs register is large enough for direct index writes
        if self._out_int_idx:
            self.outputs.resize(max(self._out_int_idx) + 1)


    def update(self, t=None):
        if self._sel_indices is not None:
            # ----------------------------------------------------------
            # Fast path: direct integer array indexing
            # ----------------------------------------------------------
            bus_array = self.inputs._data[0]
            if isinstance(bus_array, np.ndarray):
                out = self.outputs._data
                for sel_idx, out_idx in zip(self._sel_indices, self._out_int_idx):
                    out[out_idx] = bus_array[sel_idx]
        else:
            # ----------------------------------------------------------
            # Fallback: dict navigation (before prepare() is called)
            # ----------------------------------------------------------
            bus = self.inputs['bus']
            if not isinstance(bus, dict):
                return
            for key in self.keys:
                val = bus
                for part in key.split('.'):
                    if isinstance(val, dict):
                        val = val.get(part, 0.0)
                    else:
                        val = 0.0
                        break
                self.outputs[key] = val
