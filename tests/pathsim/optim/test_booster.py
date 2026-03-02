########################################################################################
##
##                                     TESTS FOR
##                                'optim/booster.py'
##
##                                Kevin McBride 2026
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np
import pytest

from pathsim.optim.booster import _collect_leaves, _flatten_bus_to_array


# HELPER TESTS =========================================================================

class TestFlattenBusToArray:

    def test_flat_dict(self):
        d = {'a': 1.0, 'b': 2.0}
        arr = _flatten_bus_to_array(d)
        np.testing.assert_array_equal(arr, [1.0, 2.0])
        assert arr.dtype == float

    def test_nested_dict(self):
        d = {'zone': {'T': 20.0, 'H': 55.0}, 'wind': 3.5}
        arr = _flatten_bus_to_array(d)
        np.testing.assert_array_equal(arr, [20.0, 55.0, 3.5])

    def test_single_key(self):
        d = {'sig': 42.0}
        np.testing.assert_array_equal(_flatten_bus_to_array(d), [42.0])

    def test_non_numeric_leaf_defaults_to_zero(self):
        d = {'a': 1.0, 'b': 'bad'}
        arr = _flatten_bus_to_array(d)
        assert arr[0] == 1.0
        assert arr[1] == 0.0

    def test_empty_dict(self):
        arr = _flatten_bus_to_array({})
        assert len(arr) == 0

    def test_deeply_nested(self):
        d = {'a': {'b': {'c': 7.0}}}
        np.testing.assert_array_equal(_flatten_bus_to_array(d), [7.0])


# INTEGRATION TESTS — bus algebraic loops =============================================
# These tests spin up a minimal Simulation to verify that ConnectionBooster
# falls back to plain FPI for bus-carrying loop-closing connections and that
# the scalar Anderson path is unaffected.

import matplotlib
matplotlib.use('Agg')

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Adder, Constant
from pathsim.blocks.buses import BusCreator, BusSelector


class TestConnectionBoosterBusFallback:

    def test_scalar_loop_still_works(self):
        """Baseline: scalar algebraic loop must still converge correctly."""
        c     = Constant(1.0)
        adder = Adder('++')
        amp   = Amplifier(0.5)
        sim = Simulation([c, adder, amp], [
            Connection(c[0],     adder[0]),
            Connection(adder[0], amp[0]),
            Connection(amp[0],   adder[1]),
        ], dt=0.1)
        sim.run(duration=0.5)
        assert abs(adder.outputs[0] - 2.0) < 1e-8

    def test_bus_in_loop_does_not_crash(self):
        """BusCreator inside a loop must not raise TypeError."""
        c2       = Constant(1.0)
        adder2   = Adder('++')
        amp2     = Amplifier(0.5)
        creator  = BusCreator(['sig'])
        selector = BusSelector(['sig'])
        sim = Simulation([c2, adder2, amp2, creator, selector], [
            Connection(c2[0],           adder2[0]),
            Connection(adder2[0],       amp2[0]),
            Connection(amp2[0],         creator['sig']),
            Connection(creator[0],      selector['bus']),
            Connection(selector['sig'], adder2[1]),
        ], dt=0.1)
        sim.run(duration=0.5)
        assert abs(adder2.outputs[0] - 2.0) < 1e-8

    def test_bus_as_loop_closing_connection(self):
        """When the loop-closing wire itself carries a dict the sim converges."""
        c3        = Constant(3.0)
        creator3  = BusCreator(['x', 'y'])
        selector3 = BusSelector(['x', 'y'])
        amp3x     = Amplifier(0.0)   # gain 0 → trivially convergent
        amp3y     = Amplifier(0.0)

        sim = Simulation([c3, creator3, selector3, amp3x, amp3y], [
            Connection(c3[0],              creator3['x']),
            Connection(amp3x[0],           creator3['y']),
            Connection(creator3[0],        selector3['bus']),
            Connection(selector3['x'],     amp3x[0]),
            Connection(selector3['y'],     amp3y[0]),
        ], dt=0.1)
        sim.run(duration=0.5)
        assert creator3.outputs['bus'] == {'x': 3.0, 'y': 0.0}

    def test_bus_loop_nested_bus(self):
        """Nested bus dict in a loop-closing connection is flattened correctly."""
        # zones_creator produces {'ZoneA': {'T': val}, 'ZoneB': {'T': val2}}
        c_a  = Constant(20.0)
        c_b  = Constant(25.0)
        cr_a = BusCreator(['T'])
        cr_b = BusCreator(['T'])
        zones = BusCreator(['ZoneA', 'ZoneB'])
        sel  = BusSelector(['ZoneA.T', 'ZoneB.T'])

        sim = Simulation([c_a, c_b, cr_a, cr_b, zones, sel], [
            Connection(c_a[0],   cr_a['T']),
            Connection(c_b[0],   cr_b['T']),
            Connection(cr_a[0],  zones['ZoneA']),
            Connection(cr_b[0],  zones['ZoneB']),
            Connection(zones[0], sel['bus']),
            # no actual feedback — loop_depth stays 0, but we test the helper
        ], dt=0.1)
        sim.run(duration=0.5)
        assert sel.outputs['ZoneA.T'] == pytest.approx(20.0)
        assert sel.outputs['ZoneB.T'] == pytest.approx(25.0)

    def test_is_bus_none_before_first_run(self):
        """_is_bus is None on all boosters before the simulation has run."""
        c     = Constant(1.0)
        adder = Adder('++')
        amp   = Amplifier(0.5)
        sim = Simulation([c, adder, amp], [
            Connection(c[0],     adder[0]),
            Connection(adder[0], amp[0]),
            Connection(amp[0],   adder[1]),
        ], dt=0.1)
        # Boosters are created at build time but update() hasn't fired yet.
        for booster in sim.boosters:
            assert booster._is_bus is None

    def test_is_bus_set_after_run(self):
        """_is_bus is not None on any booster after the simulation has run."""
        c     = Constant(1.0)
        adder = Adder('++')
        amp   = Amplifier(0.5)
        sim = Simulation([c, adder, amp], [
            Connection(c[0],     adder[0]),
            Connection(adder[0], amp[0]),
            Connection(amp[0],   adder[1]),
        ], dt=0.1)
        sim.run(duration=0.2)
        for booster in sim.boosters:
            assert booster._is_bus is not None

    def test_reset_and_rerun_converges(self):
        """A bus-in-loop simulation can be reset and re-run without error."""
        c      = Constant(1.0)
        adder  = Adder('++')
        amp    = Amplifier(0.5)
        cr     = BusCreator(['sig'])
        sel    = BusSelector(['sig'])
        sim = Simulation([c, adder, amp, cr, sel], [
            Connection(c[0],        adder[0]),
            Connection(adder[0],    amp[0]),
            Connection(amp[0],      cr['sig']),
            Connection(cr[0],       sel['bus']),
            Connection(sel['sig'],  adder[1]),
        ], dt=0.1)
        sim.run(duration=0.2)
        sim.reset()
        sim.run(duration=0.2)
        assert abs(adder.outputs[0] - 2.0) < 1e-8
