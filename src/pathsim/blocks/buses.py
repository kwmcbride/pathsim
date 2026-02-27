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


# MISO BLOCKS ===========================================================================

class BusCreator(Block):
    """Combines multiple input signals into a structured bus (dict) output.

    Takes N named scalar inputs and packs them into a single Python dict
    that is passed downstream as one signal.  The dict keys are the signal
    names; nested buses are supported by connecting the output of one
    BusCreator to a named input port of another.

    Parameters
    ----------
    keys : list[str | BusElement] | Bus
        Either a ``Bus`` object (or passed as the first positional arg) that
        defines the signal names, or a plain list of string / ``BusElement``
        names.
    bus : Bus, optional
        Deprecated positional form — prefer passing a ``Bus`` as *keys*.
    """

    def __init__(self, keys=None, bus=None, **kwargs):
        from ..bus import Bus
        if bus is not None:
            self.bus = bus
            self.keys = [e.name for e in bus.elements]
        elif isinstance(keys, Bus):
            self.bus = keys
            self.keys = [e.name for e in keys.elements]
        else:
            self.bus = None
            self.keys = [k.name if hasattr(k, 'name') else k for k in keys]
        self.input_port_labels = {k: i for i, k in enumerate(self.keys)}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs  = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)


    def update(self, t=None):
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
        self.input_port_labels  = {"bus": 0}
        self.output_port_labels = {k: i for i, k in enumerate(keys)}
        super().__init__()
        self.keys    = list(keys)
        self.inputs  = Register(mapping=self.input_port_labels,  dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)


    def update(self, t=None):
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
