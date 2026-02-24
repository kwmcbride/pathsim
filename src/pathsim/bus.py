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
    """Represents a single named signal (leaf) or nested sub-bus within a Bus.

    Parameters
    ----------
    name : str
        Name of the element, used as the port label on BusCreator / BusSelector.
    data_type : str or Bus
        Scalar data type (e.g. ``'float'``, ``'uint64'``) for leaf elements,
        or a nested :class:`Bus` instance for hierarchical buses.
    dimensions : int
        Number of dimensions for the signal (default: 1).
    unit : str, optional
        Physical unit string (e.g. ``'km/h'``, ``'kPa'``).
    description : str, optional
        Human-readable description of the signal.
    """

    def __init__(self, name, data_type='float', dimensions=1, unit=None, description=''):
        self.name = name
        self.data_type = data_type
        self.dimensions = dimensions
        self.unit = unit
        self.description = description

    def is_nested(self):
        """Return ``True`` if this element references a nested :class:`Bus`."""
        return isinstance(self.data_type, Bus)


class Bus:
    """Structured bus definition, inspired by ``Simulink.Bus``.

    A ``Bus`` groups named signals (and optionally nested ``Bus`` objects)
    into a single logical channel.  It is used to define the structure
    passed to :class:`~pathsim.blocks.buses.BusCreator` and
    :class:`~pathsim.blocks.buses.BusSelector`.

    Parameters
    ----------
    name : str
        Name of the bus (used for display and debugging).
    elements : list[BusElement], optional
        Ordered list of :class:`BusElement` objects defining the signals.
    description : str, optional
        Human-readable description of the bus.

    Attributes
    ----------
    name : str
    elements : list[BusElement]
    description : str
    """

    def __init__(self, name, elements=None, description=''):
        self.name = name
        self.description = description
        self.elements = elements if elements is not None else []
        self._element_dict = {elem.name: elem for elem in self.elements}


    def __iter__(self):
        return iter(self.elements)


    def __repr__(self):
        return f"Bus(name={self.name!r}, elements={[e.name for e in self.elements]})"


    def __getattr__(self, name):
        """Access bus elements as attributes.

        For nested elements returns the nested :class:`Bus`; for leaf
        elements returns the :class:`BusElement` itself.

        Raises ``AttributeError`` if no element with that name exists.
        """
        elem = self.get_element(name)
        if elem is None:
            raise AttributeError(f"Bus has no element '{name}'")
        return elem.data_type if elem.is_nested() else elem


    def add_element(self, element):
        """Append a :class:`BusElement` to the bus."""
        self.elements.append(element)
        self._element_dict[element.name] = element


    def get_element(self, name):
        """Return the :class:`BusElement` with *name*, or ``None`` if absent."""
        return self._element_dict.get(name, None)


    @property
    def get_element_names(self):
        """List of element names in definition order."""
        return [elem.name for elem in self.elements]


    def get_leaf_elements(self):
        """Return all non-nested (scalar) elements in depth-first order.

        Returns
        -------
        list[BusElement]
        """
        leaves = []
        for elem in self.elements:
            if elem.is_nested():
                leaves.extend(elem.data_type.get_leaf_elements())
            else:
                leaves.append(elem)
        return leaves


    def validate(self, bus_dict):
        """Check that *bus_dict* contains every top-level element.

        Parameters
        ----------
        bus_dict : dict
            Dictionary to validate against the bus structure.

        Returns
        -------
        bool
            ``True`` if valid.

        Raises
        ------
        ValueError
            If any top-level element name is missing from *bus_dict*.
        """
        for elem in self.elements:
            if elem.name not in bus_dict:
                raise ValueError(f"Missing bus element: {elem.name}")
        return True


    def structure_dict(self):
        """Return a nested dict describing the bus structure (not values).

        Returns
        -------
        dict
            Nested :class:`~collections.OrderedDict` matching the bus
            hierarchy, with leaf entries containing metadata fields
            ``data_type``, ``dimensions``, ``unit``, ``description``.
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
                    'description': elem.description,
                }
        return struct


    def pprint_structure(self):
        """Pretty-print the bus structure dict."""
        import pprint
        pprint.pprint(self.structure_dict())


    def print_tree_structure(self, indent=1, show_first_level=True):
        """Print the bus hierarchy as an indented tree.

        Parameters
        ----------
        indent : int
            Indentation level for nested buses (default: 1).
        show_first_level : bool
            Whether to print the top-level bus name line (default: ``True``).
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
                elem.data_type.print_tree_structure(indent=indent + 1, show_first_level=False)
