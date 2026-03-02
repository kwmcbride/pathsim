import warnings
import pytest
from pathsim.blocks.sources import Constant
from pathsim.blocks.buses import BusCreator, BusSelector, BusMerge
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


# BusMerge unit tests ===================================================================

def test_busmerge_basic():
    """Two non-overlapping buses are merged into one."""
    m = BusMerge(n=2)
    m.inputs['bus_0'] = {'x': 1.0, 'y': 2.0}
    m.inputs['bus_1'] = {'z': 3.0}
    m.update()
    assert m.outputs['bus'] == {'x': 1.0, 'y': 2.0, 'z': 3.0}


def test_busmerge_three_inputs():
    m = BusMerge(n=3)
    m.inputs['bus_0'] = {'a': 1.0}
    m.inputs['bus_1'] = {'b': 2.0}
    m.inputs['bus_2'] = {'c': 3.0}
    m.update()
    assert m.outputs['bus'] == {'a': 1.0, 'b': 2.0, 'c': 3.0}


def test_busmerge_conflict_last_wins():
    """Default on_conflict='warn': last bus wins, warning emitted once."""
    m = BusMerge(n=2, on_conflict='warn')
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 99.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.update()
    assert m.outputs['bus']['x'] == 99.0
    assert len(w) == 1
    assert 'x' in str(w[0].message)


def test_busmerge_conflict_warn_once():
    """Conflict warning fires only once per key, not on every update."""
    m = BusMerge(n=2, on_conflict='warn')
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 2.0}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m.update()   # first call — warns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.update()   # second call — must not warn again
    assert len(w) == 0


def test_busmerge_conflict_error():
    m = BusMerge(n=2, on_conflict='error')
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 2.0}
    with pytest.raises(ValueError, match="conflict"):
        m.update()


def test_busmerge_conflict_first():
    m = BusMerge(n=2, on_conflict='first')
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 99.0}
    m.update()
    assert m.outputs['bus']['x'] == 1.0   # first wins


def test_busmerge_conflict_last_silent():
    m = BusMerge(n=2, on_conflict='last')
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 99.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.update()
    assert m.outputs['bus']['x'] == 99.0
    assert len(w) == 0   # no warning


def test_busmerge_fpi_transient_skipped():
    """Non-dict inputs (FPI transient 0) are silently skipped."""
    m = BusMerge(n=2)
    m.inputs['bus_0'] = {'x': 5.0}
    m.inputs['bus_1'] = 0          # FPI transient
    m.update()
    assert m.outputs['bus'] == {'x': 5.0}


def test_busmerge_in_place_update():
    """Output dict identity is preserved across updates (no reallocation)."""
    m = BusMerge(n=2)
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'y': 2.0}
    m.update()
    first_id = id(m.outputs['bus'])
    m.inputs['bus_0'] = {'x': 10.0}
    m.update()
    assert id(m.outputs['bus']) == first_id
    assert m.outputs['bus'] == {'x': 10.0, 'y': 2.0}


def test_busmerge_reset_clears_warnings_and_output():
    m = BusMerge(n=2)
    m.inputs['bus_0'] = {'x': 1.0}
    m.inputs['bus_1'] = {'x': 2.0}
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        m.update()
    assert 'x' in m._warned_conflicts
    m.reset()
    assert len(m._warned_conflicts) == 0
    assert m.outputs['bus'] == 0


def test_busmerge_invalid_n():
    with pytest.raises(ValueError, match="at least 2"):
        BusMerge(n=1)


def test_busmerge_invalid_on_conflict():
    with pytest.raises(ValueError, match="on_conflict"):
        BusMerge(on_conflict='ignore')


def test_busmerge_repr():
    m = BusMerge(n=3, on_conflict='error')
    assert 'BusMerge' in repr(m)
    assert '3' in repr(m)
    assert 'error' in repr(m)
