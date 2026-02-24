from pathsim.bus import Bus, BusElement
from pathsim.blocks import BusCreator, BusSelector, Constant, Scope
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
        sensor_bus = BusCreator(keys=sensor_bus_def)
        vehicle_bus = BusCreator(keys=vehicle_bus_def)
        busselector = BusSelector(keys=['Speed', 'Sensors.Temperature'])
        buscreator = BusCreator(keys=['a', 'b'])
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
    creator = BusCreator(keys=bus_def)
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
    creator = BusCreator(keys=bus_def)
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
    assert bus.get_element_names == ['Speed', 'Status']
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

