#########################################################################################
##
##                                   BUS CLASS
##                                (pathsim/bus.py)
##
##              Provides hierarchical, structured signal grouping for blocks,
##              supporting nested buses, element metadata, and validation.
##
##                               Kevin McBride 2026
##
#########################################################################################

from collections import OrderedDict

class BusElement:
    
    """
    BusElement
    ----------
    Represents a single element (signal) in a bus.

    Parameters
    ----------
    name : str
        Name of the bus element (signal).
    data_type : str or Bus
        Data type of the element (e.g., 'float', 'uint64') or a nested Bus instance.
    dimensions : int
        Number of dimensions for the signal (default: 1).
    unit : str, optional
        Physical unit of the signal (optional).
    description : str, optional
        Description of the signal (optional).

    Attributes
    ----------
    name : str
        Element name.
    data_type : str or Bus
        Data type or nested Bus.
    dimensions : int
        Signal dimensions.
    unit : str
        Signal unit.
    description : str
        Signal description.
    """
    
    def __init__(self, name, data_type='float', dimensions=1, unit=None, description=''):
        """
        Initialize a BusElement.

        Parameters
        ----------
        name : str
            Element name.
        data_type : str or Bus
            Data type or nested Bus.
        dimensions : int
            Signal dimensions.
        unit : str, optional
            Signal unit.
        description : str, optional
            Signal description.
        """
        self.name = name
        self.data_type = data_type
        self.dimensions = dimensions
        self.unit = unit
        self.description = description

    def is_nested(self):
        """
        Return True if this element is a nested bus.
        """
        return isinstance(self.data_type, Bus)


class Bus:

    """
    Bus
    ---
    Structured bus definition, inspired by Simulink.Bus.

    Parameters
    ----------
    name : str
        Name of the bus.
    elements : list[BusElement], optional
        List of BusElement objects defining the bus structure.
    description : str, optional
        Description of the bus.

    Attributes
    ----------
    name : str
        Bus name.
    elements : list[BusElement]
        List of bus elements.
    description : str
        Bus description.
    _element_dict : dict[str, BusElement]
        Internal mapping for quick element lookup.
    """
    
    def __init__(self, name, elements=None, description=''):
        """
        Initialize a Bus.

        Parameters
        ----------
        name : str
            Bus name.
        elements : list[BusElement], optional
            List of BusElement objects.
        description : str, optional
            Bus description.
        """
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
        """
        Iterate over bus elements.
        """
        return iter(self.elements)


    def __repr__(self):
        """
        String representation of the bus.
        """
        return f"Bus(description={self.description}, elements={[e.name for e in self.elements]})"


    def __getattr__(self, name):
        """
        Access bus elements as attributes.

        Parameters
        ----------
        name : str
            Element name.

        Returns
        -------
        BusElement or Bus
            The element or nested Bus.
        """
        elem = self.get_element(name)
        if elem is None:
            raise AttributeError(f"Bus has no element '{name}'")
        # If element is nested, return the nested Bus
        if elem.is_nested():
            return elem.data_type
        return elem
    
    
    def add_element(self, element):
        """
        Add a BusElement to the bus.

        Parameters
        ----------
        element : BusElement
            Element to add.
        """
        self.elements.append(element)
        self._element_dict[element.name] = element


    def get_element(self, name):
        """
        Get a BusElement by name.

        Parameters
        ----------
        name : str
            Element name.

        Returns
        -------
        BusElement or None
            The element if found, else None.
        """
        return self._element_dict.get(name, None)


    def validate(self, bus_dict):
        """
        Validate a dict against the bus structure.

        Parameters
        ----------
        bus_dict : dict
            Dictionary to validate against the bus structure.

        Returns
        -------
        bool
            True if valid, raises ValueError otherwise.
        """
        for elem in self.elements:
            if elem.name not in bus_dict:
                raise ValueError(f"Missing bus element: {elem.name}")
            # Optionally check data_type/dimensions here
        return True


    def get_leaf_elements(self):
        """
        Return all leaf (non-nested) elements.

        Returns
        -------
        leaves : list[BusElement]
            List of leaf BusElement objects.
        """
        leaves = []
        for elem in self.elements:
            if elem.is_nested():
                leaves.extend(elem.data_type.get_leaf_elements())
            else:
                leaves.append(elem)
        return leaves
    
    
    @property
    def get_element_names(self):
        """
        Return list of element names in the bus.

        Returns
        -------
        names : list[str]
            List of element names.
        """
        return [elem.name for elem in self.elements]
    
    
    def structure_dict(self):
        """
        Return a nested dict representing the structure of the bus.

        Returns
        -------
        struct : dict
            Nested dictionary representing the bus structure.
        """
        struct = OrderedDict()
        for elem in self.elements:
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
        """
        Pretty-print the structure of the bus as a nested dict.

        Returns
        -------
        None
        """
        import pprint
        pprint.pprint(self.structure_dict())


    def print_tree_structure(self, indent=1, show_first_level=True):
        """
        Print the structure of the bus as an indented tree.

        Parameters
        ----------
        indent : int, optional
            Indentation level for nested buses (default: 1).
        show_first_level : bool, optional
            Whether to show the top-level bus name and description (default: True).

        Returns
        -------
        None
        """
        if show_first_level:
            print(f"{self.name} (Bus): {self.description}")
        prefix = '    ' * indent
        for elem in self.elements:
            if isinstance(elem.data_type, Bus):
                print(f"{prefix}{elem.name}: {elem.data_type.name} (Bus): {elem.data_type.description}")
            else:
                print(f"{prefix}{elem.name}: {elem.data_type}")
            if elem.is_nested():
                elem.data_type.print_tree_structure(indent=indent+1, show_first_level=False)
                
