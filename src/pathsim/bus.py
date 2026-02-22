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
    data_type can be a string (e.g., 'float', 'uint64') or a Bus instance (for nested buses).
    """
    def __init__(self, name, data_type='float', dimensions=1, unit=None, description=''):
        self.name = name
        self.data_type = data_type  # string or Bus instance
        self.dimensions = dimensions
        self.unit = unit
        self.description = description

    def is_nested(self):
        return isinstance(self.data_type, Bus)


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
        
        self._element_dict = {elem.name: elem for elem in self.elements}  # For quick lookup by name
        
    def add_element(self, element):
        self.elements.append(element)
        self._element_dict[element.name] = element

    def get_element(self, name):
        return self._element_dict.get(name, None)

    def validate(self, bus_dict):
        """Validate a dict against the bus structure."""
        for elem in self.elements:
            if elem.name not in bus_dict:
                raise ValueError(f"Missing bus element: {elem.name}")
            # Optionally check data_type/dimensions here
        return True

    def get_leaf_elements(self):
        """Return all leaf (non-nested) elements."""
        leaves = []
        for elem in self.elements:
            if elem.is_nested():
                leaves.extend(elem.data_type.get_leaf_elements())
            else:
                leaves.append(elem)
        return leaves

    def __iter__(self):
        return iter(self.elements)

    def __repr__(self):
        return f"Bus(description={self.description}, elements={[e.name for e in self.elements]})"

    def __getattr__(self, name):
        elem = self.get_element(name)
        if elem is None:
            raise AttributeError(f"Bus has no element '{name}'")
        # If element is nested, return the nested Bus
        if elem.is_nested():
            return elem.data_type
        return elem
    
    
    @property
    def get_element_names(self):
        """Return list of element names in the bus."""
        return [elem.name for elem in self.elements]


# Example usage:
# chirp = BusElement('Chirp', data_type='float')
# sine = BusElement('Sine', data_type='float')
# sinusoidal = Bus([chirp, sine], description='Sinusoidal signals')
# nested = BusElement('NestedBus', data_type=sinusoidal)
# step = BusElement('Step', data_type='float')
# top_bus = Bus([nested, step], description='Top-level bus')
