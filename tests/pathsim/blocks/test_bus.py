import pytest
from pathsim.blocks.sources import Constant
from pathsim.blocks.buses import BusCreator, BusSelector
from pathsim.blocks.scope import Scope

# Minimal simulation harness for block wiring
def test_bus_creator_and_selector():
    # Create two constant blocks
    c1 = Constant(1.23)
    c2 = Constant(4.56)

    # Update constants to set their outputs
    c1.update(0)
    c2.update(0)

    # Bus creator with two keys
    bus_creator = BusCreator(['a', 'b'])

    # Bus selector to select only 'a'
    bus_selector = BusSelector(['a'])

    # Scope to observe output
    scope = Scope()


    # Use string keys for Register assignment (now supported)
    bus_creator.inputs['a'] = c1.outputs['out']
    bus_creator.inputs['b'] = c2.outputs['out']
    bus_creator.update()

    bus_selector.inputs['bus'] = bus_creator.outputs['bus']
    bus_selector.update()

    scope.inputs[0] = bus_selector.outputs['a']
    scope.update(0)

    # Check that the bus creator output is correct
    assert bus_creator.outputs['bus'] == {'a': 1.23, 'b': 4.56}
    # Check that the selector output is correct
    assert bus_selector.outputs['a'] == 1.23
    # Check that the scope input is correct
    assert scope.inputs[0] == 1.23
