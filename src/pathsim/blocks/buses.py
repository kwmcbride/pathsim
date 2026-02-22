#########################################################################################
##
##                             BUS CREATOR AND SELECTOR BLOCKS 
##                                   (blocks/bus.py)
##
##                                   Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block
from ..connection import Connection
from ..utils.register import Register


# MISO BLOCKS ===========================================================================

class BusCreator(Block):
    
    """
    Combines multiple input signals into a bus (dict) output.
    Args:
        bus (Bus): Bus object defining the bus structure.
    Output:
        outputs['bus']: dict with keys mapping to input values
    """
    
    def __init__(self, keys=None, bus=None, **kwargs):
        if bus is not None:
            self.bus = bus
            self.keys = [e.name for e in bus.elements]
        else:
            self.bus = None
            self.keys = [k.name if hasattr(k, 'name') else k for k in keys]
        self.input_port_labels = {k: i for i, k in enumerate(self.keys)}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        # Always use dtype=object for bus signals
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)


    def update(self, t=None):
        if hasattr(self, 'bus') and self.bus is not None:
            bus_dict = {e.name: self.inputs[e.name] for e in self.bus.elements}
        else:
            bus_dict = {}
            for k in self.keys:
                key_str = k.name if hasattr(k, 'name') else k
                parts = key_str.split('.')
                d = bus_dict
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        d[part] = self.inputs[key_str]
                    else:
                        if part not in d:
                            d[part] = {}
                        d = d[part]
        self.outputs['bus'] = bus_dict


class BusSelector(Block):
    
    """
    Selects one or more keys from a bus (dict) input and outputs them as separate outputs.
    Args:
        keys (list): List of keys to select from the bus.
    Output:
        outputs[key]: value from bus[key]
    """
    
    def __init__(self, keys):
        self.input_port_labels = {"bus": 0}
        self.output_port_labels = {k: i for i, k in enumerate(keys)}
        super().__init__()
        self.keys = list(keys)
        # Always use dtype=object for bus signals
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)


    def update(self, t=None):
        bus = self.inputs['bus']
        for i, key in enumerate(self.keys):
            val = bus
            for part in key.split('.'):
                if isinstance(val, dict):
                    val = val.get(part, 0.0)
                else:
                    val = 0.0
            self.outputs[key] = val

# I might need this in the future for more complex bus handling, but for now I'll keep it simple and just use the keys directly in BusConnection
# class BusConnection(Connection):
#     def __init__(self, source, target, bus_structure):
#         super().__init__(source, target)
#         self.bus_structure = bus_structure
    