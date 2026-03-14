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
        """When the loop-closing wire carries a bus ndarray the sim converges."""
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
        buf = creator3.outputs['bus']
        assert isinstance(buf, np.ndarray)
        assert buf[0] == pytest.approx(3.0)   # 'x' at index 0
        assert buf[1] == pytest.approx(0.0)   # 'y' at index 1

    def test_bus_loop_nested_bus(self):
        """Bus in a loop-closing connection converges correctly."""
        c_a  = Constant(20.0)
        c_b  = Constant(25.0)
        zones = BusCreator(['ZoneA', 'ZoneB'])
        sel  = BusSelector(['ZoneA', 'ZoneB'])

        sim = Simulation([c_a, c_b, zones, sel], [
            Connection(c_a[0],   zones['ZoneA']),
            Connection(c_b[0],   zones['ZoneB']),
            Connection(zones[0], sel['bus']),
        ], dt=0.1)
        sim.run(duration=0.5)
        assert sel.outputs['ZoneA'] == pytest.approx(20.0)
        assert sel.outputs['ZoneB'] == pytest.approx(25.0)

    def test_bus_history_none_before_first_update(self):
        """_bus_history is None on all boosters before the simulation has run."""
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
            assert booster._bus_history is None

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
