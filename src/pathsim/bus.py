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

import warnings
from collections import OrderedDict


_VALID_DATA_TYPES = frozenset({
    'float', 'int',
    'uint8', 'uint16', 'uint32', 'uint64',
    'int8', 'int16', 'int32', 'int64',
    'bool', 'complex',
})


class BusElement:
    """Represents a single named signal (leaf) or nested sub-bus within a Bus.

    Parameters
    ----------
    name : str
        Name of the element, used as the port label on BusCreator / BusSelector.
    data_type : str or Bus
        Scalar data type (e.g. ``'float'``, ``'uint64'``) for leaf elements,
        or a nested :class:`Bus` instance for hierarchical buses.
        Valid string values: ``'float'``, ``'int'``, ``'bool'``, ``'complex'``,
        ``'uint8'``, ``'uint16'``, ``'uint32'``, ``'uint64'``,
        ``'int8'``, ``'int16'``, ``'int32'``, ``'int64'``.
    dimensions : int
        Number of dimensions for the signal (default: 1).
    unit : str, optional
        Physical unit string (e.g. ``'km/h'``, ``'kPa'``). Defaults to ``''``.
    description : str, optional
        Human-readable description of the signal. Defaults to ``''``.
    """

    def __init__(self, name, data_type='float', dimensions=1, unit='', description=''):
        self.name = name
        # Validate data_type: must be a known type string or a Bus instance.
        # Bus is defined later in this same module; by call time it is available.
        if not (isinstance(data_type, str) and data_type in _VALID_DATA_TYPES):
            if not isinstance(data_type, Bus):
                warnings.warn(
                    f"BusElement '{name}': unrecognised data_type {data_type!r}. "
                    f"Expected one of {sorted(_VALID_DATA_TYPES)} or a Bus instance.",
                    UserWarning,
                    stacklevel=2,
                )
        self.data_type = data_type
        self.dimensions = dimensions
        self.unit = unit
        self.description = description

    def is_nested(self):
        """Return ``True`` if this element references a nested :class:`Bus`."""
        return isinstance(self.data_type, Bus)

    def __eq__(self, other):
        if not isinstance(other, BusElement):
            return NotImplemented
        return (
            self.name == other.name
            and self.data_type == other.data_type
            and self.dimensions == other.dimensions
            and self.unit == other.unit
            and self.description == other.description
        )

    def __hash__(self):
        # data_type may be a Bus (unhashable if it contains mutable lists),
        # so fall back to id-based hashing for nested elements.
        try:
            return hash((self.name, self.data_type, self.dimensions, self.unit, self.description))
        except TypeError:
            return hash((self.name, id(self.data_type), self.dimensions, self.unit, self.description))


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
        self.elements = []
        self._element_dict = {}
        for elem in (elements or []):
            self.add_element(elem)


    def __iter__(self):
        return iter(self.elements)


    def __repr__(self):
        return f"Bus(name={self.name!r}, elements={[e.name for e in self.elements]})"

    def __eq__(self, other):
        if not isinstance(other, Bus):
            return NotImplemented
        return (
            self.name == other.name
            and self.description == other.description
            and self.elements == other.elements
        )

    def __hash__(self):
        # Buses are mutable (add_element), so hash by identity.
        return id(self)


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
        """Append a :class:`BusElement` to the bus.

        Raises
        ------
        ValueError
            If an element with the same name already exists on this bus.
        """
        if element.name in self._element_dict:
            raise ValueError(
                f"Bus '{self.name}' already has an element named '{element.name}'."
            )
        self.elements.append(element)
        self._element_dict[element.name] = element


    def get_element(self, name):
        """Return the :class:`BusElement` with *name*, or ``None`` if absent."""
        return self._element_dict.get(name, None)


    @property
    def element_names(self):
        """List of element names in definition order."""
        return [elem.name for elem in self.elements]

    @property
    def get_element_names(self):
        """Deprecated. Use :attr:`element_names` instead."""
        warnings.warn(
            "Bus.get_element_names is deprecated; use Bus.element_names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.element_names


    def get_leaf_elements(self, _visited=None):
        """Return all non-nested (scalar) elements in depth-first order.

        Parameters
        ----------
        _visited : set, optional
            Internal parameter used for cycle detection. Do not pass explicitly.

        Returns
        -------
        list[BusElement]

        Raises
        ------
        ValueError
            If a circular reference between ``Bus`` objects is detected.
        """
        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            raise ValueError(
                f"Circular reference detected in Bus '{self.name}'. "
                f"Bus definitions must not contain cycles."
            )
        _visited.add(id(self))
        leaves = []
        for elem in self.elements:
            if elem.is_nested():
                leaves.extend(elem.data_type.get_leaf_elements(_visited))
            else:
                leaves.append(elem)
        return leaves


    def validate(self, bus_dict, _path=''):
        """Check that *bus_dict* matches the bus structure at all levels.

        Parameters
        ----------
        bus_dict : dict
            Dictionary to validate against the bus structure.
        _path : str, optional
            Internal parameter tracking the key path for error messages.
            Do not pass explicitly.

        Returns
        -------
        bool
            ``True`` if valid.

        Raises
        ------
        ValueError
            If any element name is missing, or a nested element receives a
            non-dict value.
        """
        prefix = f"{_path}." if _path else ''
        for elem in self.elements:
            full_key = f"{prefix}{elem.name}"
            if elem.name not in bus_dict:
                raise ValueError(f"Missing bus element: '{full_key}'")
            if elem.is_nested():
                value = bus_dict[elem.name]
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Bus element '{full_key}' is a nested Bus but received "
                        f"{type(value).__name__!r} instead of a dict."
                    )
                elem.data_type.validate(value, _path=full_key)
        return True


    def structure_dict(self, _visited=None):
        """Return a nested dict describing the bus structure (not values).

        Parameters
        ----------
        _visited : set, optional
            Internal parameter used for cycle detection. Do not pass explicitly.

        Returns
        -------
        dict
            Nested :class:`~collections.OrderedDict` matching the bus
            hierarchy, with leaf entries containing metadata fields
            ``data_type``, ``dimensions``, ``unit``, ``description``.

        Raises
        ------
        ValueError
            If a circular reference between ``Bus`` objects is detected.
        """
        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            raise ValueError(
                f"Circular reference detected in Bus '{self.name}'. "
                f"Bus definitions must not contain cycles."
            )
        _visited.add(id(self))
        struct = OrderedDict()
        for elem in self.elements:
            if elem.is_nested():
                struct[elem.name] = elem.data_type.structure_dict(_visited)
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


    def print_tree_structure(self, indent=1, show_first_level=True, _visited=None):
        """Print the bus hierarchy as an indented tree.

        Parameters
        ----------
        indent : int
            Indentation level for nested buses (default: 1).
        show_first_level : bool
            Whether to print the top-level bus name line (default: ``True``).
        _visited : set, optional
            Internal parameter used for cycle detection. Do not pass explicitly.

        Raises
        ------
        ValueError
            If a circular reference between ``Bus`` objects is detected.
        """
        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            raise ValueError(
                f"Circular reference detected in Bus '{self.name}'. "
                f"Bus definitions must not contain cycles."
            )
        _visited.add(id(self))
        if show_first_level:
            print(f"{self.name} (Bus): {self.description}")
        prefix = '    ' * indent
        for elem in self.elements:
            if isinstance(elem.data_type, Bus):
                print(f"{prefix}{elem.name}: {elem.data_type.name} (Bus): {elem.data_type.description}")
            else:
                print(f"{prefix}{elem.name}: {elem.data_type}")
            if elem.is_nested():
                elem.data_type.print_tree_structure(
                    indent=indent + 1, show_first_level=False, _visited=_visited
                )
