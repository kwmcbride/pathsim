import warnings

from pathsim.bus import Bus, BusElement
from pathsim.blocks import BusCreator, BusSelector, BusMerge, BusFunction, Constant, Scope
from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.solvers import SSPRK22

import pytest

# TESTS ==========================================================================================


class BusSimulation:
    """
    Dummy simulation class to test bus wiring and Scope output, using the provided simulation setup.
    """
    def __init__(self):
        # Define nested sensor bus
        sensor_temp = BusElement('Temperature', data_type='float', unit='C', description='Ambient temperature')
        sensor_pressure = BusElement('Pressure', data_type='float', unit='kPa', description='Atmospheric pressure')
        sensor_bus_def = Bus('bus_sensor', elements=[sensor_temp, sensor_pressure], description='Sensor signals')

        # Define top-level vehicle bus
        speed_elem = BusElement('Speed', data_type='float', unit='km/h', description='Vehicle speed')
        status_elem = BusElement('Status', data_type='uint64', description='Vehicle status')
        sensors_elem = BusElement('Sensors', data_type=sensor_bus_def)
        vehicle_bus_def = Bus('bus_vehicle', elements=[speed_elem, status_elem, sensors_elem], description='Vehicle bus')

        # Sources
        temperature = Constant(value=22)
        pressure = Constant(value=101)
        status = Constant()
        speed = Constant(value=55)

        # Algebraic blocks
        sensor_bus = BusCreator(sensor_bus_def)
        vehicle_bus = BusCreator(vehicle_bus_def)
        busselector = BusSelector(keys=['Speed', 'Sensors.Temperature'])
        buscreator = BusCreator(['a', 'b'])
        block_8 = BusSelector(keys=['b'])

        # Recording
        scope1 = Scope()
        scope2 = Scope()

        self.blocks = [
            temperature,
            pressure,
            status,
            speed,
            sensor_bus,
            vehicle_bus,
            busselector,
            buscreator,
            block_8,
            scope1,
            scope2,
        ]

        self.connections = [
            Connection(speed[0], vehicle_bus[0]),
            Connection(status[0], vehicle_bus[1], buscreator[0]),
            Connection(pressure[0], sensor_bus[1], buscreator[1]),
            Connection(sensor_bus[0], vehicle_bus[2]),
            Connection(vehicle_bus[0], busselector[0]),
            Connection(busselector[0], scope1[0]),
            Connection(busselector[1], scope1[1]),
            Connection(temperature[0], sensor_bus[0]),
            Connection(buscreator[0], block_8[0]),
            Connection(block_8[0], scope2[0]),
        ]

        self.scope1 = scope1
        self.scope2 = scope2
        
        self.sim = Simulation(
            self.blocks,
            self.connections,
            Solver=SSPRK22,
            dt=0.01,
            dt_min=1e-16,
            tolerance_lte_rel=0.0001,
            tolerance_lte_abs=1e-08,
            tolerance_fpi=1e-10,
        )

    def run(self, duration=1.0):
        self.sim.run(duration=duration)
        return {'scope1': self.scope1.read(), 'scope2': self.scope2.read()}


def test_bus_simulation_scope_output():
    sim = BusSimulation()
    res = sim.run(duration=1.0)
    
    # y should have two channels: Speed and Sensors.Temperature
    t1, y1 = res['scope1']
    assert y1.shape[0] == 2, f"Scope output should have 2 channels, got {y1.shape[0]}"
    speed_channel = y1[0]
    temp_channel = y1[1]
    
    # Check that the values are correct for all timesteps
    for val in speed_channel:
        assert val == 55, f"Speed channel value incorrect: {val}"
    for val in temp_channel:
        assert val == 22, f"Temperature channel value incorrect: {val}"
        
    # Check that the simple array bus definition is working correctly
    t2, y2 = res['scope2']
    pressure_channel = y2[0]
    for val in pressure_channel:
        assert val == 101, f"Scope2 pressure value incorrect: {val}"


def test_bus_through_subsystem():
    """Bus produced inside a Subsystem should reach an external BusSelector."""
    bus_def = Bus('b', elements=[BusElement('x'), BusElement('y')])

    # Inner subsystem: packs two Constants into a bus
    iface = Interface()
    c1 = Constant(3.0)
    c2 = Constant(7.0)
    creator = BusCreator(bus_def)
    sub_blocks = [iface, c1, c2, creator]
    sub_connections = [
        Connection(c1[0], creator['x']),
        Connection(c2[0], creator['y']),
        Connection(creator[0], iface[0]),  # bus → subsystem output
    ]
    inner = Subsystem(sub_blocks, sub_connections)

    # Outer level: selector picks both signals
    selector = BusSelector(keys=['x', 'y'])
    scope = Scope()

    blocks = [inner, selector, scope]
    connections = [
        Connection(inner[0], selector[0]),
        Connection(selector['x'], scope[0]),
        Connection(selector['y'], scope[1]),
    ]

    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 2
    assert all(v == 3.0 for v in y[0]), f"x channel wrong: {y[0]}"
    assert all(v == 7.0 for v in y[1]), f"y channel wrong: {y[1]}"


def test_bus_through_nested_subsystems():
    """Bus produced inside a doubly-nested Subsystem should reach an external BusSelector."""
    bus_def = Bus('b', elements=[BusElement('a'), BusElement('b')])

    # Innermost subsystem: packs two Constants into a bus
    iface_inner = Interface()
    c1 = Constant(10.0)
    c2 = Constant(20.0)
    creator = BusCreator(bus_def)
    inner_blocks = [iface_inner, c1, c2, creator]
    inner_connections = [
        Connection(c1[0], creator['a']),
        Connection(c2[0], creator['b']),
        Connection(creator[0], iface_inner[0]),
    ]
    inner = Subsystem(inner_blocks, inner_connections)

    # Middle subsystem: just passes the bus through
    iface_mid = Interface()
    mid_blocks = [iface_mid, inner]
    mid_connections = [
        Connection(inner[0], iface_mid[0]),
    ]
    mid = Subsystem(mid_blocks, mid_connections)

    # Outer level: selector reads from doubly-nested bus
    selector = BusSelector(keys=['a', 'b'])
    scope = Scope()

    blocks = [mid, selector, scope]
    connections = [
        Connection(mid[0], selector[0]),
        Connection(selector['a'], scope[0]),
        Connection(selector['b'], scope[1]),
    ]

    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 2
    assert all(v == 10.0 for v in y[0]), f"a channel wrong: {y[0]}"
    assert all(v == 20.0 for v in y[1]), f"b channel wrong: {y[1]}"


def test_bus_element_basic():
    elem = BusElement('Speed', data_type='float', dimensions=1, unit='km/h', description='Vehicle speed')
    assert elem.name == 'Speed'
    assert elem.data_type == 'float'
    assert elem.dimensions == 1
    assert elem.unit == 'km/h'
    assert elem.description == 'Vehicle speed'
    assert not elem.is_nested()


def test_bus_flat():
    speed = BusElement('Speed', data_type='float')
    status = BusElement('Status', data_type='uint64')
    bus = Bus('bus_vehicle', elements=[speed, status], description='Vehicle bus')
    assert bus.get_element('Speed') == speed
    assert bus.get_element('Status') == status
    assert bus.get_element('Missing') is None
    assert bus.element_names == ['Speed', 'Status']
    assert bus.validate({'Speed': 1.0, 'Status': 2})
    assert bus.get_leaf_elements() == [speed, status]


def test_bus_nested():
    temp = BusElement('Temperature', data_type='float')
    pressure = BusElement('Pressure', data_type='float')
    sensor_bus = Bus('bus_sensor', elements=[temp, pressure], description='Sensor signals')
    sensors = BusElement('Sensors', data_type=sensor_bus)
    speed = BusElement('Speed', data_type='float')
    bus = Bus('bus_vehicle', elements=[speed, sensors], description='Vehicle bus')
    assert sensors.is_nested()
    assert bus.get_element('Sensors') == sensors
    assert bus.get_element('Speed') == speed
    # Nested attribute access
    assert bus.Sensors == sensor_bus
    assert sensor_bus.Temperature == temp
    # Leaf elements
    leaves = bus.get_leaf_elements()
    assert temp in leaves and speed in leaves and pressure in leaves


def test_structure_dict_and_pprint(capsys):
    temp = BusElement('Temperature', data_type='float')
    pressure = BusElement('Pressure', data_type='float')
    sensor_bus = Bus('bus_sensor', elements=[temp, pressure], description='Sensor signals')
    sensors = BusElement('Sensors', data_type=sensor_bus)
    speed = BusElement('Speed', data_type='float')
    bus = Bus('bus_vehicle', elements=[speed, sensors], description='Vehicle bus')
    struct = bus.structure_dict()
    assert 'Speed' in struct and 'Sensors' in struct
    assert 'Temperature' in struct['Sensors'] and 'Pressure' in struct['Sensors']
    bus.pprint_structure()
    out = capsys.readouterr().out
    assert 'Speed' in out and 'Sensors' in out and 'Temperature' in out


def test_print_tree_structure(capsys):
    temp = BusElement('Temperature', data_type='float')
    pressure = BusElement('Pressure', data_type='float')
    sensor_bus = Bus('bus_sensor', elements=[temp, pressure], description='Sensor signals')
    sensors = BusElement('Sensors', data_type=sensor_bus)
    speed = BusElement('Speed', data_type='float')
    bus = Bus('bus_vehicle', elements=[speed, sensors], description='Vehicle bus')
    bus.print_tree_structure()
    out = capsys.readouterr().out
    assert 'bus_vehicle' in out and 'Speed' in out and 'Sensors' in out and 'Temperature' in out and 'Pressure' in out


def test_add_element_and_repr():
    bus = Bus('bus_test', description='Test bus')
    elem = BusElement('A', data_type='float')
    bus.add_element(elem)
    assert bus.get_element('A') == elem
    assert 'A' in bus.__repr__()


def test_invalid_element_access():
    bus = Bus('bus_test', description='Test bus')
    with pytest.raises(AttributeError):
        _ = bus.Missing


def test_validate_missing_element():
    bus = Bus('bus_test', elements=[BusElement('A', data_type='float')], description='Test bus')
    with pytest.raises(ValueError):
        bus.validate({'B': 1.0})


# =============================================================================
# Critical fix tests
# =============================================================================

def test_bus_element_unit_default_is_empty_string():
    elem = BusElement('x')
    assert elem.unit == ''
    assert elem.description == ''


def test_bus_creator_repr_with_bus():
    bus_def = Bus('sensors', elements=[BusElement('x'), BusElement('y')])
    creator = BusCreator(bus_def)
    r = repr(creator)
    assert 'BusCreator' in r and 'sensors' in r and 'x' in r


def test_bus_creator_repr_plain_keys():
    creator = BusCreator(['a', 'b'])
    r = repr(creator)
    assert 'BusCreator' in r and 'a' in r and 'b' in r


def test_bus_selector_repr():
    selector = BusSelector(keys=['Speed', 'Sensors.Temp'])
    r = repr(selector)
    assert 'BusSelector' in r and 'Speed' in r and 'Sensors.Temp' in r


def test_bus_element_equality():
    e1 = BusElement('Speed', data_type='float', unit='km/h')
    e2 = BusElement('Speed', data_type='float', unit='km/h')
    assert e1 == e2


def test_bus_element_inequality():
    e1 = BusElement('Speed', data_type='float', unit='km/h')
    e2 = BusElement('Speed', data_type='float', unit='mph')
    assert e1 != e2


def test_bus_equality():
    b1 = Bus('vehicle', elements=[BusElement('Speed'), BusElement('Status', data_type='int')])
    b2 = Bus('vehicle', elements=[BusElement('Speed'), BusElement('Status', data_type='int')])
    assert b1 == b2


def test_bus_inequality_name():
    b1 = Bus('a', elements=[BusElement('x')])
    b2 = Bus('b', elements=[BusElement('x')])
    assert b1 != b2


def test_bus_inequality_elements():
    b1 = Bus('bus', elements=[BusElement('x')])
    b2 = Bus('bus', elements=[BusElement('y')])
    assert b1 != b2


def test_bus_nested_equality():
    inner1 = Bus('inner', elements=[BusElement('x'), BusElement('y')])
    inner2 = Bus('inner', elements=[BusElement('x'), BusElement('y')])
    e1 = BusElement('sub', data_type=inner1)
    e2 = BusElement('sub', data_type=inner2)
    assert e1 == e2


def test_bus_creator_reset_clears_output():
    import numpy as np
    creator = BusCreator(['x', 'y'])
    creator.inputs['x'] = 1.0
    creator.inputs['y'] = 2.0
    creator.update()
    # BusCreator now outputs a flat float64 ndarray (not a dict).
    assert isinstance(creator.outputs['bus'], np.ndarray)
    creator.reset()
    assert creator.outputs['bus'] == 0


def test_bus_selector_reset_clears_output_and_warnings(pathsim_logs):
    selector = BusSelector(keys=['a', 'missing'])
    selector.inputs['bus'] = {'a': 5.0}
    selector.update()
    assert any('missing' in r.message for r in pathsim_logs)
    assert 'missing' in selector._warned_missing
    selector.reset()
    # After reset: outputs zeroed, warning state cleared
    assert selector.outputs['a'] == 0
    assert len(selector._warned_missing) == 0


def test_bus_element_valid_data_type():
    # Known types should not warn
    for dtype in ('float', 'int', 'uint64', 'bool', 'complex'):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BusElement('x', data_type=dtype)


def test_bus_element_bus_data_type_no_warn():
    inner = Bus('inner', elements=[BusElement('a')])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BusElement('nested', data_type=inner)


def test_bus_element_invalid_data_type_warns():
    with pytest.warns(UserWarning, match="unrecognised data_type"):
        BusElement('x', data_type='FancyType')


def test_bus_element_non_string_data_type_warns():
    with pytest.warns(UserWarning, match="unrecognised data_type"):
        BusElement('x', data_type=42)


def test_validate_recursive_nested_bus():
    inner = Bus('inner', elements=[BusElement('x'), BusElement('y')])
    outer = Bus('outer', elements=[BusElement('val'), BusElement('inner', data_type=inner)])
    # Valid nested structure
    assert outer.validate({'val': 1.0, 'inner': {'x': 2.0, 'y': 3.0}})


def test_validate_nested_receives_scalar():
    inner = Bus('inner', elements=[BusElement('x')])
    outer = Bus('outer', elements=[BusElement('inner', data_type=inner)])
    with pytest.raises(ValueError, match="nested Bus"):
        outer.validate({'inner': 42.0})


def test_validate_nested_missing_key():
    inner = Bus('inner', elements=[BusElement('x'), BusElement('y')])
    outer = Bus('outer', elements=[BusElement('inner', data_type=inner)])
    with pytest.raises(ValueError, match="inner.y"):
        outer.validate({'inner': {'x': 1.0}})  # 'y' missing


def test_bus_creator_duplicate_keys():
    with pytest.raises(ValueError):
        BusCreator(['x', 'y', 'x'])


def test_bus_creator_empty_list_raises():
    with pytest.raises(ValueError, match="at least one element"):
        BusCreator([])


def test_bus_creator_invalid_type_raises():
    with pytest.raises(TypeError, match="Bus object or a list"):
        BusCreator(42)


def test_bus_selector_duplicate_keys():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        BusSelector(keys=['a', 'b', 'a'])


def test_duplicate_element_name_in_add_element():
    bus = Bus('b', elements=[BusElement('Speed')])
    with pytest.raises(ValueError, match="Speed"):
        bus.add_element(BusElement('Speed'))


def test_duplicate_element_name_in_init():
    with pytest.raises(ValueError, match="Speed"):
        Bus('b', elements=[BusElement('Speed'), BusElement('Speed')])


def test_element_names_property():
    speed = BusElement('Speed')
    status = BusElement('Status')
    bus = Bus('b', elements=[speed, status])
    assert bus.element_names == ['Speed', 'Status']


def test_get_element_names_deprecated():
    bus = Bus('b', elements=[BusElement('A')])
    with pytest.warns(DeprecationWarning, match="element_names"):
        names = bus.get_element_names
    assert names == ['A']


def test_bus_connection_alias():
    """BusConnection must be importable from pathsim and behave like Connection."""
    from pathsim import BusConnection, Connection
    assert BusConnection is Connection


def test_bus_connection_works_in_simulation():
    """PathView-generated BusConnection wires should run without error."""
    from pathsim import Simulation, BusConnection
    c1 = Constant(1.0)
    c2 = Constant(2.0)
    creator = BusCreator(['a', 'b'])
    selector = BusSelector(keys=['a', 'b'])
    scope = Scope()
    blocks = [c1, c2, creator, selector, scope]
    connections = [
        Connection(c1[0], creator['a']),
        Connection(c2[0], creator['b']),
        BusConnection(creator[0], selector[0]),
        Connection(selector['a'], scope[0]),
        Connection(selector['b'], scope[1]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)
    t, y = scope.read()
    assert all(v == 1.0 for v in y[0])
    assert all(v == 2.0 for v in y[1])


def test_bus_selector_warns_on_missing_key(pathsim_logs):
    """BusSelector should log when a requested key is absent."""
    selector = BusSelector(keys=['x', 'missing_key'])
    selector.inputs['bus'] = {'x': 42.0}
    selector.update()
    assert any('missing_key' in r.message for r in pathsim_logs)
    assert selector.outputs['x'] == 42.0
    assert selector.outputs['missing_key'] == 0.0


def test_bus_selector_warns_on_missing_key_only_once(pathsim_logs):
    """Missing-key message should fire only once per key, not every timestep."""
    selector = BusSelector(keys=['ghost'])
    selector.inputs['bus'] = {'x': 1.0}
    selector.update()
    assert len([r for r in pathsim_logs if 'BUS WARNING' in r.message]) == 1
    selector.update()  # second call — must not add another record
    assert len([r for r in pathsim_logs if 'BUS WARNING' in r.message]) == 1


def test_bus_selector_warns_on_missing_nested_key(pathsim_logs):
    """Dot-notation key miss should log the full key name."""
    selector = BusSelector(keys=['Sensors.Temp'])
    selector.inputs['bus'] = {'Sensors': {'Pressure': 101.0}}
    selector.update()
    assert any('Sensors.Temp' in r.message for r in pathsim_logs)
    assert selector.outputs['Sensors.Temp'] == 0.0


def test_bus_selector_warns_on_non_dict_input(pathsim_warnings):
    """BusSelector should warn when the bus input is not a dict (and not the FPI zero)."""
    selector = BusSelector(keys=['x'])
    selector.inputs['bus'] = 'wrong_type'
    selector.update()
    assert any('non-dict' in r.message for r in pathsim_warnings)


def test_bus_selector_silent_on_fpi_zero(pathsim_warnings):
    """BusSelector must NOT warn when bus is 0 (the FPI initial state)."""
    selector = BusSelector(keys=['x'])
    selector.inputs['bus'] = 0  # FPI transient
    selector.update()
    assert len(pathsim_warnings) == 0


def test_bus_circular_reference_get_leaf_elements():
    """Circular Bus references must raise ValueError in get_leaf_elements."""
    bus_a = Bus('a', elements=[])
    bus_b = Bus('b', elements=[])
    bus_a.add_element(BusElement('child', data_type=bus_b))
    bus_b.add_element(BusElement('child', data_type=bus_a))
    with pytest.raises(ValueError, match="[Cc]ircular"):
        bus_a.get_leaf_elements()


def test_bus_circular_reference_structure_dict():
    """Circular Bus references must raise ValueError in structure_dict."""
    bus_a = Bus('a', elements=[])
    bus_b = Bus('b', elements=[])
    bus_a.add_element(BusElement('child', data_type=bus_b))
    bus_b.add_element(BusElement('child', data_type=bus_a))
    with pytest.raises(ValueError, match="[Cc]ircular"):
        bus_a.structure_dict()


def test_bus_circular_reference_print_tree():
    """Circular Bus references must raise ValueError in print_tree_structure."""
    bus_a = Bus('a', elements=[])
    bus_b = Bus('b', elements=[])
    bus_a.add_element(BusElement('child', data_type=bus_b))
    bus_b.add_element(BusElement('child', data_type=bus_a))
    with pytest.raises(ValueError, match="[Cc]ircular"):
        bus_a.print_tree_structure()


def test_bus_self_reference():
    """A Bus containing itself must raise ValueError."""
    bus = Bus('self_ref', elements=[])
    bus.add_element(BusElement('me', data_type=bus))
    with pytest.raises(ValueError, match="[Cc]ircular"):
        bus.get_leaf_elements()


# =============================================================================
# Bus-aware Scope tests (F1)
# =============================================================================

def test_scope_explicit_bus_labels():
    """Scope(bus=) derives channel labels from Bus element names and units."""
    bus_def = Bus('sensors', elements=[
        BusElement('Temperature', unit='C'),
        BusElement('Pressure', unit='kPa'),
    ])
    scope = Scope(bus=bus_def)
    assert scope.labels == ['Temperature [C]', 'Pressure [kPa]']


def test_scope_explicit_bus_labels_no_unit():
    """Elements without a unit should appear without the unit bracket."""
    bus_def = Bus('b', elements=[BusElement('x'), BusElement('y')])
    scope = Scope(bus=bus_def)
    assert scope.labels == ['x', 'y']


def test_scope_explicit_bus_user_labels_not_overridden():
    """When labels= is passed explicitly, bus= should not override them."""
    bus_def = Bus('b', elements=[BusElement('x'), BusElement('y')])
    scope = Scope(bus=bus_def, labels=['my_x', 'my_y'])
    assert scope.labels == ['my_x', 'my_y']


def test_scope_bus_direct_connection_flat():
    """Scope with bus= records leaf scalars when a bus dict is connected directly."""
    bus_def = Bus('b', elements=[BusElement('x'), BusElement('y')])
    c1 = Constant(3.0)
    c2 = Constant(7.0)
    creator = BusCreator(bus_def)
    scope = Scope(bus=bus_def)

    blocks = [c1, c2, creator, scope]
    connections = [
        Connection(c1[0], creator['x']),
        Connection(c2[0], creator['y']),
        Connection(creator[0], scope[0]),   # bus direct to Scope
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 2, f"Expected 2 channels, got {y.shape[0]}"
    assert all(v == 3.0 for v in y[0]), f"x channel: {y[0]}"
    assert all(v == 7.0 for v in y[1]), f"y channel: {y[1]}"


def test_scope_bus_direct_connection_nested():
    """Scope with nested bus= expands to all leaf elements in depth-first order."""
    inner = Bus('inner', elements=[BusElement('a'), BusElement('b')])
    outer = Bus('outer', elements=[
        BusElement('val'),
        BusElement('inner', data_type=inner),
    ])

    c_val = Constant(1.0)
    c_a   = Constant(2.0)
    c_b   = Constant(3.0)
    inner_creator = BusCreator(inner)
    outer_creator = BusCreator(outer)
    scope = Scope(bus=outer)

    blocks = [c_val, c_a, c_b, inner_creator, outer_creator, scope]
    connections = [
        Connection(c_a[0],   inner_creator['a']),
        Connection(c_b[0],   inner_creator['b']),
        Connection(c_val[0], outer_creator['val']),
        Connection(inner_creator[0], outer_creator['inner']),
        Connection(outer_creator[0], scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    # Depth-first: val, inner.a, inner.b
    assert y.shape[0] == 3, f"Expected 3 channels, got {y.shape[0]}"
    assert all(v == 1.0 for v in y[0]), f"val: {y[0]}"
    assert all(v == 2.0 for v in y[1]), f"inner.a: {y[1]}"
    assert all(v == 3.0 for v in y[2]), f"inner.b: {y[2]}"


def test_scope_lazy_bus_detection():
    """Plain Scope() auto-detects a bus dict and expands channels lazily."""
    bus_def = Bus('b', elements=[BusElement('p'), BusElement('q')])
    c1 = Constant(10.0)
    c2 = Constant(20.0)
    creator = BusCreator(bus_def)
    scope = Scope()   # no bus= — lazy detection

    blocks = [c1, c2, creator, scope]
    connections = [
        Connection(c1[0], creator['p']),
        Connection(c2[0], creator['q']),
        Connection(creator[0], scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 2, f"Expected 2 channels after lazy detection, got {y.shape[0]}"
    assert all(v == 10.0 for v in y[0]), f"p: {y[0]}"
    assert all(v == 20.0 for v in y[1]), f"q: {y[1]}"


def test_scope_lazy_auto_labels():
    """Lazy detection sets self.labels from the dict key paths."""
    bus_def = Bus('b', elements=[BusElement('alpha'), BusElement('beta')])
    creator = BusCreator(bus_def)
    c1 = Constant(1.0)
    c2 = Constant(2.0)
    scope = Scope()

    blocks = [c1, c2, creator, scope]
    connections = [
        Connection(c1[0], creator['alpha']),
        Connection(c2[0], creator['beta']),
        Connection(creator[0], scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    assert 'alpha' in scope.labels
    assert 'beta' in scope.labels


def test_busmerge_simulation_two_buses():
    """BusMerge combines two buses end-to-end through a Simulation."""
    bus_a = Bus('a', elements=[BusElement('x'), BusElement('y')])
    bus_b = Bus('b', elements=[BusElement('z')])

    ca1 = Constant(1.0)
    ca2 = Constant(2.0)
    cb1 = Constant(3.0)
    creator_a = BusCreator(bus_a)
    creator_b = BusCreator(bus_b)
    merger    = BusMerge(n=2)
    selector  = BusSelector(keys=['x', 'y', 'z'])
    scope     = Scope()

    blocks = [ca1, ca2, cb1, creator_a, creator_b, merger, selector, scope]
    connections = [
        Connection(ca1[0], creator_a['x']),
        Connection(ca2[0], creator_a['y']),
        Connection(cb1[0], creator_b['z']),
        Connection(creator_a[0], merger['bus_0']),
        Connection(creator_b[0], merger['bus_1']),
        Connection(merger[0],    selector['bus']),
        Connection(selector['x'], scope[0]),
        Connection(selector['y'], scope[1]),
        Connection(selector['z'], scope[2]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 3
    assert all(v == 1.0 for v in y[0]), f"x: {y[0]}"
    assert all(v == 2.0 for v in y[1]), f"y: {y[1]}"
    assert all(v == 3.0 for v in y[2]), f"z: {y[2]}"


def test_busmerge_simulation_direct_scope():
    """BusMerge output fed directly into a bus-aware Scope."""
    bus_a = Bus('a', elements=[BusElement('p'), BusElement('q')])
    bus_b = Bus('b', elements=[BusElement('r')])
    merged_bus = Bus('merged', elements=[BusElement('p'), BusElement('q'), BusElement('r')])

    c1, c2, c3 = Constant(10.0), Constant(20.0), Constant(30.0)
    creator_a = BusCreator(bus_a)
    creator_b = BusCreator(bus_b)
    merger    = BusMerge(n=2)
    scope     = Scope(bus=merged_bus)

    blocks = [c1, c2, c3, creator_a, creator_b, merger, scope]
    connections = [
        Connection(c1[0], creator_a['p']),
        Connection(c2[0], creator_a['q']),
        Connection(c3[0], creator_b['r']),
        Connection(creator_a[0], merger['bus_0']),
        Connection(creator_b[0], merger['bus_1']),
        Connection(merger[0],    scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)

    t, y = scope.read()
    assert y.shape[0] == 3
    assert all(v == 10.0 for v in y[0])
    assert all(v == 20.0 for v in y[1])
    assert all(v == 30.0 for v in y[2])


def test_scope_bus_resets_correctly():
    """After reset(), Scope with bus= records fresh data on second run."""
    bus_def = Bus('b', elements=[BusElement('x'), BusElement('y')])
    c1 = Constant(5.0)
    c2 = Constant(6.0)
    creator = BusCreator(bus_def)
    scope = Scope(bus=bus_def)

    blocks = [c1, c2, creator, scope]
    connections = [
        Connection(c1[0], creator['x']),
        Connection(c2[0], creator['y']),
        Connection(creator[0], scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)
    sim.run(duration=0.5)
    t1, y1 = scope.read()

    sim.run(duration=0.5, reset=True)
    t2, y2 = scope.read()

    assert y1.shape == y2.shape
    assert all(v == 5.0 for v in y2[0])
    assert all(v == 6.0 for v in y2[1])


# BUS SCHEMA VALIDATION TESTS ============================================================
#
# _check_bus_schemas() fires a logger WARNING for direct BusCreator → BusSelector
# connections where the selector requests keys absent from the creator's schema.
#
# PathSim's 'pathsim' root logger has propagate=False, so pytest's caplog fixture
# (which hooks the root logger) cannot intercept it.  We use a local fixture that
# adds a handler directly to the 'pathsim' logger.


def _make_zone_bus():
    return Bus('Zone', elements=[
        BusElement('Temperature', data_type='float', unit='C'),
        BusElement('Humidity',    data_type='float', unit='%RH'),
    ])


def test_schema_valid_no_warning(pathsim_logs):
    """Valid BusCreator → BusSelector: no BUS WARNING emitted."""
    zone_bus = _make_zone_bus()
    c1 = Constant(20.0); c2 = Constant(55.0)
    creator  = BusCreator(zone_bus)
    selector = BusSelector(['Temperature', 'Humidity'])
    sim = Simulation([c1, c2, creator, selector], [
        Connection(c1[0],      creator['Temperature']),
        Connection(c2[0],      creator['Humidity']),
        Connection(creator[0], selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    bus_warns = [r for r in pathsim_logs if 'BUS WARNING' in r.message]
    assert len(bus_warns) == 0


def test_schema_missing_key_warns(pathsim_logs):
    """Selector requests a key not in the bus — static BUS WARNING is emitted."""
    zone_bus = _make_zone_bus()
    c1 = Constant(20.0); c2 = Constant(55.0)
    creator  = BusCreator(zone_bus)
    selector = BusSelector(['Temperature', 'WindSpeed'])   # WindSpeed absent
    sim = Simulation([c1, c2, creator, selector], [
        Connection(c1[0],      creator['Temperature']),
        Connection(c2[0],      creator['Humidity']),
        Connection(creator[0], selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    # Static-check message contains "schema"; runtime BusSelector warning does not.
    static_warns = [r for r in pathsim_logs if 'BUS WARNING' in r.message and 'schema' in r.message]
    assert len(static_warns) >= 1
    assert 'WindSpeed' in static_warns[0].message


def test_schema_all_missing_warns_once(pathsim_logs):
    """Multiple missing keys are reported in a single static BUS WARNING."""
    zone_bus = _make_zone_bus()
    c1 = Constant(20.0); c2 = Constant(55.0)
    creator  = BusCreator(zone_bus)
    selector = BusSelector(['WindSpeed', 'Pressure'])
    sim = Simulation([c1, c2, creator, selector], [
        Connection(c1[0],      creator['Temperature']),
        Connection(c2[0],      creator['Humidity']),
        Connection(creator[0], selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    static_warns = [r for r in pathsim_logs
                    if 'BUS WARNING' in r.message and 'schema' in r.message]
    assert len(static_warns) >= 1
    assert 'WindSpeed' in static_warns[0].message
    assert 'Pressure' in static_warns[0].message


def test_schema_plain_keys_top_level_check(pathsim_logs):
    """BusCreator with plain string keys: top-level missing key warns."""
    c1 = Constant(1.0); c2 = Constant(2.0)
    creator  = BusCreator(['a', 'b'])
    selector = BusSelector(['a', 'c'])   # 'c' not in ['a', 'b']
    sim = Simulation([c1, c2, creator, selector], [
        Connection(c1[0],      creator['a']),
        Connection(c2[0],      creator['b']),
        Connection(creator[0], selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    static_warns = [r for r in pathsim_logs
                    if 'BUS WARNING' in r.message and 'schema' in r.message and 'c' in r.message]
    assert len(static_warns) >= 1


def test_schema_nested_valid_no_warning(pathsim_logs):
    """Valid dot-path key into a nested bus: no warning."""
    zone_bus = _make_zone_bus()
    system_bus = Bus('System', elements=[
        BusElement('Zone',    data_type=zone_bus),
        BusElement('Outdoor', data_type='float', unit='C'),
    ])
    c1 = Constant(20.0); c2 = Constant(55.0); c3 = Constant(8.0)
    cr_zone = BusCreator(zone_bus)
    cr_sys  = BusCreator(system_bus)
    selector = BusSelector(['Zone.Temperature', 'Outdoor'])
    sim = Simulation([c1, c2, c3, cr_zone, cr_sys, selector], [
        Connection(c1[0],       cr_zone['Temperature']),
        Connection(c2[0],       cr_zone['Humidity']),
        Connection(cr_zone[0],  cr_sys['Zone']),
        Connection(c3[0],       cr_sys['Outdoor']),
        Connection(cr_sys[0],   selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    bus_warns = [r for r in pathsim_logs if 'BUS WARNING' in r.message]
    assert len(bus_warns) == 0


def test_schema_nested_invalid_key_warns(pathsim_logs):
    """Invalid dot-path into a nested bus: static BUS WARNING emitted."""
    zone_bus = _make_zone_bus()
    system_bus = Bus('System', elements=[
        BusElement('Zone',    data_type=zone_bus),
        BusElement('Outdoor', data_type='float', unit='C'),
    ])
    c1 = Constant(20.0); c2 = Constant(55.0); c3 = Constant(8.0)
    cr_zone = BusCreator(zone_bus)
    cr_sys  = BusCreator(system_bus)
    selector = BusSelector(['Zone.Temperature', 'Zone.WindSpeed'])  # WindSpeed invalid
    sim = Simulation([c1, c2, c3, cr_zone, cr_sys, selector], [
        Connection(c1[0],       cr_zone['Temperature']),
        Connection(c2[0],       cr_zone['Humidity']),
        Connection(cr_zone[0],  cr_sys['Zone']),
        Connection(c3[0],       cr_sys['Outdoor']),
        Connection(cr_sys[0],   selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    static_warns = [r for r in pathsim_logs
                    if 'BUS WARNING' in r.message and 'schema' in r.message and 'WindSpeed' in r.message]
    assert len(static_warns) >= 1


def test_schema_check_through_busmerge(pathsim_logs):
    """BusCreator → BusMerge → BusSelector: compile step now traces through BusMerge."""
    zone_bus = _make_zone_bus()
    c1 = Constant(20.0); c2 = Constant(55.0)
    creator  = BusCreator(zone_bus)
    merger   = BusMerge(n=2)
    selector = BusSelector(['Temperature', 'WindSpeed'])  # 'WindSpeed' missing in schema
    sim = Simulation([c1, c2, creator, merger, selector], [
        Connection(c1[0],      creator['Temperature']),
        Connection(c2[0],      creator['Humidity']),
        Connection(creator[0], merger['bus_0']),
        Connection(merger[0],  selector['bus']),
    ], dt=0.1)
    sim.run(duration=0.1)
    # Compile step now traces through BusMerge and warns about the missing key.
    static_warns = [r for r in pathsim_logs
                    if 'BUS WARNING' in r.message and 'WindSpeed' in r.message]
    assert len(static_warns) >= 1


def test_schema_mismatch_through_subsystem(pathsim_logs):
    """BusCreator → Subsystem(passthrough) → BusSelector: schema check crosses Subsystem boundary."""
    zone_bus = _make_zone_bus()  # keys: Temperature, Humidity
    c1 = Constant(20.0); c2 = Constant(55.0)
    creator  = BusCreator(zone_bus)

    # Build a passthrough subsystem: bus enters at port 0, exits at port 0.
    # Interface output 0 (= signal from outside) wired directly to Interface input 0
    # (= signal going to outside).
    iface = Interface()
    passthrough = Subsystem(
        blocks=[iface],
        connections=[Connection(iface[0], iface[0])],
    )

    selector = BusSelector(['Temperature', 'WindSpeed'])  # 'WindSpeed' is invalid

    sim = Simulation(
        [c1, c2, creator, passthrough, selector],
        [
            Connection(c1[0],          creator['Temperature']),
            Connection(c2[0],          creator['Humidity']),
            Connection(creator[0],     passthrough[0]),
            Connection(passthrough[0], selector['bus']),
        ],
        dt=0.1,
    )
    sim.run(duration=0.1)
    static_warns = [r for r in pathsim_logs
                    if 'BUS WARNING' in r.message and 'schema' in r.message and 'WindSpeed' in r.message]
    assert len(static_warns) >= 1


# BUS FUNCTION TESTS ======================================================================

class TestBusFunction:

    # --- Constructor validation ---------------------------------------------------------

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            BusFunction(42, ['a'], ['b'])

    def test_empty_in_keys_raises(self):
        with pytest.raises(ValueError, match="in_keys"):
            BusFunction(lambda: None, [], ['b'])

    def test_empty_out_keys_raises(self):
        with pytest.raises(ValueError, match="out_keys"):
            BusFunction(lambda a: a, ['a'], [])

    def test_duplicate_in_key_raises(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            BusFunction(lambda a, b: (a, b), ['x', 'x'], ['p', 'q'])

    def test_duplicate_out_key_raises(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            BusFunction(lambda a: (a, a), ['x'], ['p', 'p'])

    # --- repr ---------------------------------------------------------------------------

    def test_repr_named_function(self):
        def my_transform(T): return T * 2
        bf = BusFunction(my_transform, ['Temperature'], ['Temperature_x2'])
        r = repr(bf)
        assert 'BusFunction' in r
        assert 'my_transform' in r
        assert 'Temperature' in r

    def test_repr_lambda(self):
        bf = BusFunction(lambda T: T, ['T'], ['T_out'])
        r = repr(bf)
        assert 'BusFunction' in r

    # --- update mechanics ---------------------------------------------------------------

    def test_single_key_transform(self):
        bf = BusFunction(lambda T: T * 1.8 + 32.0, ['Temperature'], ['Temperature_F'])
        bf.inputs['bus'] = {'Temperature': 100.0}
        bf.update()
        result = bf.outputs['bus']
        assert isinstance(result, dict)
        assert abs(result['Temperature_F'] - 212.0) < 1e-9

    def test_multi_key_transform(self):
        bf = BusFunction(
            lambda T, H: (T + 1.0, H * 2.0),
            ['T', 'H'],
            ['T_out', 'H_out'],
        )
        bf.inputs['bus'] = {'T': 10.0, 'H': 5.0}
        bf.update()
        out = bf.outputs['bus']
        assert abs(out['T_out'] - 11.0) < 1e-9
        assert abs(out['H_out'] - 10.0) < 1e-9

    def test_missing_key_defaults_to_zero(self, pathsim_logs):
        bf = BusFunction(lambda x: x * 2, ['missing'], ['out'])
        bf.inputs['bus'] = {'a': 1.0}
        bf.update()
        assert any('missing' in r.message for r in pathsim_logs)
        assert bf.outputs['bus']['out'] == 0.0

    def test_missing_key_warns_only_once(self, pathsim_logs):
        bf = BusFunction(lambda x: x, ['ghost'], ['out'])
        bf.inputs['bus'] = {'a': 1.0}
        bf.update()
        bus_warns = [r for r in pathsim_logs if 'BUS WARNING' in r.message]
        assert len(bus_warns) == 1
        bf.update()  # second call — must not add another record
        assert len([r for r in pathsim_logs if 'BUS WARNING' in r.message]) == 1

    def test_fpi_zero_silent(self, pathsim_warnings):
        """BusFunction must not warn when bus is 0 (FPI initial state)."""
        bf = BusFunction(lambda x: x, ['a'], ['out'])
        bf.inputs['bus'] = 0
        bf.update()
        assert len(pathsim_warnings) == 0
        assert bf.outputs['bus'] == 0  # output unchanged

    def test_non_dict_warns(self, pathsim_warnings):
        bf = BusFunction(lambda x: x, ['a'], ['out'])
        bf.inputs['bus'] = 'wrong'
        bf.update()
        assert any('non-dict' in r.message for r in pathsim_warnings)

    def test_dot_notation_in_key(self):
        """Dot-notation extracts values from nested bus dicts."""
        bf = BusFunction(
            lambda T: T * 1.8 + 32.0,
            ['Sensors.Temperature'],
            ['Temperature_F'],
        )
        bf.inputs['bus'] = {'Sensors': {'Temperature': 0.0}}
        bf.update()
        assert abs(bf.outputs['bus']['Temperature_F'] - 32.0) < 1e-9

    def test_output_updated_in_place(self):
        """Second call updates the dict in place (no new allocation)."""
        bf = BusFunction(lambda x: x * 2, ['v'], ['v2'])
        bf.inputs['bus'] = {'v': 3.0}
        bf.update()
        first_dict = bf.outputs['bus']
        bf.inputs['bus'] = {'v': 5.0}
        bf.update()
        # Same dict object — updated in place.
        assert bf.outputs['bus'] is first_dict
        assert abs(bf.outputs['bus']['v2'] - 10.0) < 1e-9

    # --- reset ----------------------------------------------------------

    def test_reset_clears_warnings(self, pathsim_logs):
        bf = BusFunction(lambda x: x, ['ghost'], ['out'])
        bf.inputs['bus'] = {'a': 1.0}
        bf.update()
        assert len([r for r in pathsim_logs if 'BUS WARNING' in r.message]) == 1
        assert 'ghost' in bf._warned_missing
        bf.reset()
        assert len(bf._warned_missing) == 0

    def test_reset_clears_output(self):
        bf = BusFunction(lambda x: x, ['a'], ['out'])
        bf.inputs['bus'] = {'a': 7.0}
        bf.update()
        assert isinstance(bf.outputs['bus'], dict)
        bf.reset()
        assert bf.outputs['bus'] == 0

    # --- integration (Simulation) -------------------------------------------------------

    def test_in_simulation_basic(self):
        """BusFunction inside a Simulation produces correct bus output."""
        c_T = Constant(100.0)
        c_H = Constant(50.0)
        creator = BusCreator(['Temperature', 'Humidity'])
        bf = BusFunction(
            lambda T, H: (T * 1.8 + 32.0, H / 100.0),
            in_keys=['Temperature', 'Humidity'],
            out_keys=['Temperature_F', 'Humidity_fraction'],
        )
        selector = BusSelector(['Temperature_F', 'Humidity_fraction'])
        scope = Scope()

        blocks = [c_T, c_H, creator, bf, selector, scope]
        connections = [
            Connection(c_T[0],    creator['Temperature']),
            Connection(c_H[0],    creator['Humidity']),
            Connection(creator[0], bf['bus']),
            Connection(bf[0],      selector['bus']),
            Connection(selector['Temperature_F'],       scope[0]),
            Connection(selector['Humidity_fraction'],   scope[1]),
        ]
        sim = Simulation(blocks, connections, dt=0.1)
        sim.run(duration=0.5)

        t, y = scope.read()
        assert y.shape[0] == 2
        assert all(abs(v - 212.0) < 1e-6 for v in y[0]), f"Temperature_F: {y[0]}"
        assert all(abs(v - 0.5)   < 1e-6 for v in y[1]), f"Humidity_fraction: {y[1]}"

    def test_in_simulation_scope_bus_direct(self):
        """BusFunction output connected directly to a bus-aware Scope."""
        c = Constant(20.0)
        creator = BusCreator(['T'])
        bf = BusFunction(lambda T: T + 273.15, in_keys=['T'], out_keys=['T_K'])
        out_bus = Bus('result', elements=[BusElement('T_K', unit='K')])
        scope = Scope(bus=out_bus)

        blocks = [c, creator, bf, scope]
        connections = [
            Connection(c[0],       creator['T']),
            Connection(creator[0], bf['bus']),
            Connection(bf[0],      scope[0]),
        ]
        sim = Simulation(blocks, connections, dt=0.1)
        sim.run(duration=0.5)

        t, y = scope.read()
        assert y.shape[0] == 1
        assert all(abs(v - 293.15) < 1e-6 for v in y[0])

    def test_chained_busfunctions(self):
        """Two BusFunctions in series compose correctly."""
        c = Constant(0.0)
        creator = BusCreator(['T_C'])
        to_kelvin = BusFunction(lambda T: T + 273.15, ['T_C'], ['T_K'])
        to_rankine = BusFunction(lambda T: T * 9.0 / 5.0, ['T_K'], ['T_R'])
        selector = BusSelector(['T_R'])
        scope = Scope()

        blocks = [c, creator, to_kelvin, to_rankine, selector, scope]
        connections = [
            Connection(c[0],          creator['T_C']),
            Connection(creator[0],    to_kelvin['bus']),
            Connection(to_kelvin[0],  to_rankine['bus']),
            Connection(to_rankine[0], selector['bus']),
            Connection(selector['T_R'], scope[0]),
        ]
        sim = Simulation(blocks, connections, dt=0.1)
        sim.run(duration=0.5)

        t, y = scope.read()
        # 0°C → 273.15 K → 491.67 R
        assert all(abs(v - 491.67) < 1e-3 for v in y[0]), f"T_R: {y[0]}"

    def test_reset_and_rerun(self):
        """BusFunction can be reset and re-run without error."""
        c = Constant(5.0)
        creator = BusCreator(['x'])
        bf = BusFunction(lambda x: x ** 2, ['x'], ['x2'])
        selector = BusSelector(['x2'])
        scope = Scope()

        blocks = [c, creator, bf, selector, scope]
        connections = [
            Connection(c[0],       creator['x']),
            Connection(creator[0], bf['bus']),
            Connection(bf[0],      selector['bus']),
            Connection(selector['x2'], scope[0]),
        ]
        sim = Simulation(blocks, connections, dt=0.1)
        sim.run(duration=0.3)
        sim.run(duration=0.3, reset=True)

        t, y = scope.read()
        assert all(abs(v - 25.0) < 1e-6 for v in y[0])

    def test_busfunction_output_into_busmerge(self):
        """BusFunction output merged with a second BusCreator via BusMerge."""
        c_raw = Constant(100.0)
        creator = BusCreator(['raw'])
        to_scaled = BusFunction(lambda x: x * 0.01, ['raw'], ['scaled'])

        c_extra = Constant(42.0)
        creator2 = BusCreator(['extra'])

        merger = BusMerge(n=2)
        selector = BusSelector(['scaled', 'extra'])
        scope = Scope()

        blocks = [c_raw, creator, to_scaled, c_extra, creator2, merger, selector, scope]
        connections = [
            Connection(c_raw[0],       creator['raw']),
            Connection(creator[0],     to_scaled['bus']),
            Connection(c_extra[0],     creator2['extra']),
            Connection(to_scaled[0],   merger['bus_0']),
            Connection(creator2[0],    merger['bus_1']),
            Connection(merger[0],      selector['bus']),
            Connection(selector['scaled'], scope[0]),
            Connection(selector['extra'],  scope[1]),
        ]
        sim = Simulation(blocks, connections, dt=0.1)
        sim.run(duration=0.3)

        t, y = scope.read()
        assert all(abs(v - 1.0)  < 1e-9 for v in y[0]), f"scaled: {y[0]}"
        assert all(abs(v - 42.0) < 1e-9 for v in y[1]), f"extra: {y[1]}"


# SUBSYSTEM + INTERFACE PASSTHROUGH TESTS =============================================

def test_bus_into_subsystem_with_internal_selector():
    """External BusCreator → Subsystem with internal BusSelector → external Scope.

    The inner BusSelector must have _flat_indices injected by _build_bus_layout_subsystem
    so the fast ndarray path works.  Because the Subsystem has len==0 (Interface is
    non-algebraic), it is placed at graph depth 0 and processes *before* the outer
    BusCreator at depth 1 on the very first evaluation.  The t=0 recording is therefore
    0; all subsequent recordings are correct.
    """
    bus_def = Bus('env', elements=[BusElement('T'), BusElement('H')])

    c_T = Constant(25.0)
    c_H = Constant(60.0)
    creator = BusCreator(bus_def)

    # Inner subsystem: receives bus on port 0, picks 'T' with a BusSelector,
    # and exposes it via iface output 0.
    iface = Interface()
    inner_selector = BusSelector(['T'])
    inner_blocks = [iface, inner_selector]
    inner_connections = [
        Connection(iface[0],            inner_selector['bus']),
        Connection(inner_selector['T'], iface[0]),
    ]
    sub = Subsystem(inner_blocks, inner_connections)

    scope = Scope()
    blocks = [c_T, c_H, creator, sub, scope]
    connections = [
        Connection(c_T[0],    creator['T']),
        Connection(c_H[0],    creator['H']),
        Connection(creator[0], sub[0]),
        Connection(sub[0],     scope[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)

    # Verify flat indices were injected at compile time.
    import numpy as np
    assert inner_selector._flat_indices is not None
    assert inner_selector._flat_indices[0] == 0  # 'T' is leaf 0 in bus_def

    sim.run(duration=0.3)

    t, y = scope.read()
    # Skip t=0 recording (depth-0 sub runs before depth-1 creator on first eval).
    assert all(abs(v - 25.0) < 1e-9 for v in y[0][1:]), f"T: {y[0]}"


def test_bus_passthrough_subsystem_ndarray():
    """Bus ndarray passes through a Subsystem passthrough; selector gets _flat_indices.

    Because the passthrough Subsystem has len==0, it is placed at depth 0 in the
    outer graph and processes before BusCreator at depth 1 on the first evaluation.
    The t=0 recording is therefore 0; all subsequent recordings are correct.
    """
    bus_def = Bus('sig', elements=[BusElement('a'), BusElement('b')])

    c_a = Constant(7.0)
    c_b = Constant(13.0)
    creator = BusCreator(bus_def)

    iface = Interface()
    sub = Subsystem(blocks=[iface], connections=[Connection(iface[0], iface[0])])

    selector = BusSelector(['a', 'b'])
    scope = Scope()

    blocks = [c_a, c_b, creator, sub, selector, scope]
    connections = [
        Connection(c_a[0],     creator['a']),
        Connection(c_b[0],     creator['b']),
        Connection(creator[0], sub[0]),
        Connection(sub[0],     selector['bus']),
        Connection(selector['a'], scope[0]),
        Connection(selector['b'], scope[1]),
    ]
    sim = Simulation(blocks, connections, dt=0.1)

    # Verify compile step injected flat indices via _get_subsystem_output_map passthrough.
    import numpy as np
    assert selector._flat_indices is not None
    np.testing.assert_array_equal(selector._flat_indices, [0, 1])

    sim.run(duration=0.3)

    t, y = scope.read()
    # Skip t=0 (depth-0 ordering issue); check steady-state values.
    assert all(abs(v - 7.0)  < 1e-9 for v in y[0][1:]), f"a: {y[0]}"
    assert all(abs(v - 13.0) < 1e-9 for v in y[1][1:]), f"b: {y[1]}"
