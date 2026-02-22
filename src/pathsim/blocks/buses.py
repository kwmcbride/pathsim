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

class BusStructure:
    def __init__(self, keys):
        self.keys = list(keys)
    def to_array(self, bus_dict):
        return [bus_dict[k] for k in self.keys]
    def from_array(self, arr):
        return dict(zip(self.keys, arr))
    

class BusConnection(Connection):
    def __init__(self, source, target, bus_structure):
        super().__init__(source, target)
        self.bus_structure = bus_structure
    


class BusCreator(Block):
    """
    Combines multiple input signals into a bus (dict) output.
    Args:
        bus (Bus): Bus object defining the bus structure.
    Output:
        outputs['bus']: dict with keys mapping to input values
    """
    def __init__(self, keys=None, bus=None, **kwargs):
        # Accept flat or dot notation keys
        if bus is not None:
            self.bus = bus
            # Use element names for keys
            self.keys = [e.name for e in bus.elements]
        else:
            self.bus = None
            # Convert keys to strings if they are BusElement
            self.keys = [k.name if hasattr(k, 'name') else k for k in keys]
        self.input_port_labels = {k: i for i, k in enumerate(self.keys)}
        self.output_port_labels = {"bus": 0}
        super().__init__()
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        # Debug: print dtype for each input port
        print("BusCreator input dtype:", self.inputs._data.dtype)
        for k, idx in self.input_port_labels.items():
            print(f"Port '{k}' (index {idx}): dtype={self.inputs._data.dtype}")

    def update(self, t=None):
        def assemble_bus(bus, inputs, keys):
            bus_dict = {}
            for idx, elem in enumerate(bus.elements):
                key = elem.name
                value = inputs[key] if key in inputs else inputs[idx]
                # If element is a nested bus, always assign input dict if present
                if hasattr(elem, 'nested_bus') and elem.nested_bus is not None:
                    if isinstance(value, dict):
                        bus_dict[key] = value
                    else:
                        # Recursively assemble nested bus
                        bus_dict[key] = assemble_bus(elem.nested_bus, inputs, keys)
                else:
                    bus_dict[key] = value
            return bus_dict
        # Use self.bus if available, else fallback to flat dict
        if hasattr(self, 'bus') and self.bus is not None:
            # Inputs may be indexed or keyed
            inputs = {}
            for idx, k in enumerate(self.keys):
                key_str = k.name if hasattr(k, 'name') else k
                inputs[key_str] = self.inputs[key_str]
                inputs[idx] = self.inputs[key_str]
            bus_dict = assemble_bus(self.bus, inputs, self.keys)
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
        print(f"[DEBUG] BusCreator outputs['bus']: {self.outputs['bus']}")


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
        self.inputs = Register(mapping=self.input_port_labels, dtype=object)
        self.outputs = Register(mapping=self.output_port_labels, dtype=object)
        
        
    def update(self, t=None):
        bus = self.inputs['bus']
        print(f"[DEBUG] {self.__class__.__name__} inputs['bus']: {bus}")
        for i, key in enumerate(self.keys):
            # Support dot notation for nested keys
            val = bus
            for part in key.split('.'):
                if isinstance(val, dict):
                    val = val.get(part, 0.0)
                else:
                    val = 0.0
            self.outputs[key] = val
            print(f"[DEBUG] BusSelector extracted '{key}': {val}")
        print(f"[DEBUG] {self.__class__.__name__} outputs: {self.outputs._data}")


# Example usage (for test or documentation):
"""
from pathsim.blocks.constant import Constant
from pathsim.blocks.bus import BusCreator, BusSelector
from pathsim.blocks.scope import Scope

# Two constant blocks
c1 = Constant(1.23)
c2 = Constant(4.56)

# Bus creator with two keys
bus_creator = BusCreator(['a', 'b'])

# Bus selector to select only 'a'
bus_selector = BusSelector(['a'])

# Scope to observe output
scope = Scope()

# Connect blocks (pseudo-code, depends on your simulation framework)
c1.outputs[0] -> bus_creator.inputs['a']
c2.outputs[0] -> bus_creator.inputs['b']
bus_creator.outputs['bus'] -> bus_selector.inputs['bus']
bus_selector.outputs['a'] -> scope.inputs[0]
"""

#########################################################################################
##
##                                   BUS CLASS (bus.py)
##
##              Python equivalent of Simulink.Bus for structured signals
##
#########################################################################################

class BusElement:
    """
    Represents a single element (signal) in a bus.
    """
    def __init__(self, name, dtype='float', dimensions=1, description='', nested_bus=None):
        self.name = name
        self.dtype = dtype
        self.dimensions = dimensions
        self.description = description
        self.nested_bus = nested_bus  # For hierarchical/nested buses

    def is_nested(self):
        return self.nested_bus is not None

class Bus:
    """
    Structured bus definition, inspired by Simulink.Bus.
    - elements: list of BusElement
    - description: string
    - supports nested buses
    """
    def __init__(self, elements=None, description='', data_scope='auto', header_file='', alignment=-1, preserve_dims=False):
        self.description = description
        self.elements = elements if elements is not None else []
        self.data_scope = data_scope
        self.header_file = header_file
        self.alignment = alignment
        self.preserve_dims = preserve_dims

    def add_element(self, element):
        self.elements.append(element)

    def get_element(self, name):
        for elem in self.elements:
            if elem.name == name:
                return elem
        return None

    def validate(self, bus_dict):
        """Validate a dict against the bus structure."""
        for elem in self.elements:
            if elem.name not in bus_dict:
                raise ValueError(f"Missing bus element: {elem.name}")
            # Optionally check dtype/dimensions here
        return True

    def get_leaf_elements(self):
        """Return all leaf (non-nested) elements."""
        leaves = []
        for elem in self.elements:
            if elem.is_nested():
                leaves.extend(elem.nested_bus.get_leaf_elements())
            else:
                leaves.append(elem)
        return leaves

    def __repr__(self):
        return f"Bus(description={self.description}, elements={[e.name for e in self.elements]})"

# Example usage:
# chirp = BusElement('Chirp')
# sine = BusElement('Sine')
# sinusoidal = Bus([chirp, sine], description='Sinusoidal signals')
# nested = BusElement('NestedBus', dtype='bus', nested_bus=sinusoidal)
# step = BusElement('Step')
# top_bus = Bus([nested, step], description='Top-level bus')