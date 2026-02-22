#########################################################################################
##
##                                   BUS CLASS (bus.py)
##
##              Python equivalent of Simulink.Bus for structured signals
##
#########################################################################################

from collections import OrderedDict

class BusElement:
    
    """
    Represents a single element (signal) in a bus.
    data_type can be a string (e.g., 'float', 'uint64') or a Bus instance (for nested buses).
    """
    
    def __init__(self, name, data_type='float', dimensions=1, unit=None, description=''):
        self.name = name
        self.data_type = data_type
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
    
    def __init__(self, name, elements=None, description=''): # , data_scope='auto', header_file='', alignment=-1, preserve_dims=False):
        self.name = name
        self.description = description
        self.elements = elements if elements is not None else []
        # These are only for code generation metadata, not used in current implementation but stored for future use
        # self.data_scope = data_scope
        # self.header_file = header_file
        # self.alignment = alignment
        # self.preserve_dims = preserve_dims
        
        self._element_dict = {elem.name: elem for elem in self.elements}  # For quick lookup by name
        
    
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
    
    
    @property
    def get_element_names(self):
        """Return list of element names in the bus."""
        return [elem.name for elem in self.elements]
    
    
    def structure_dict(self):
        """Return a nested dict representing the structure of the bus."""
        struct = OrderedDict()
        for elem in self.elements:
            # print(f"Processing element '{elem.name}' of type '{elem.data_type}'")
            if elem.is_nested():
                struct[elem.name] = elem.data_type.structure_dict()
            else:
                struct[elem.name] = {
                    'data_type': elem.data_type,
                    'dimensions': elem.dimensions,
                    'unit': elem.unit,
                    'description': elem.description
                }
   
        return struct


    def pprint_structure(self):
        """Pretty-print the structure of the bus as a nested dict."""
        import pprint
        pprint.pprint(self.structure_dict())


    def print_tree_structure(self, indent=1, show_first_level=True):
        """Print the structure of the bus as an indented tree."""
        if show_first_level:
            print(f"{self.name} (Bus): {self.description}")
        prefix = '    ' * indent
        for elem in self.elements:
            if isinstance(elem.data_type, Bus):
                print(f"{prefix}{elem.name}: {elem.data_type.name} (Bus): {elem.data_type.description}")
            else:
                print(f"{prefix}{elem.name}: {elem.data_type}")# [{elem.dimensions}D], {elem.unit if elem.unit else '-'} | {elem.description}")
            if elem.is_nested():
                elem.data_type.print_tree_structure(indent=indent+1, show_first_level=False)
