########################################################################################
##
##                                   TESTS FOR 
##                          'pathsim.events.schedule.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events.schedule import (
    Schedule,
    ScheduleList
    )


# TESTS ================================================================================

class TestSchedule(unittest.TestCase):
    """
    Test the implementation of the 'Schedule' event class.
    """

    def test_init(self):

        S = Schedule(
            t_start=0.1, 
            t_end=200, 
            t_period=20 
            )

        self.assertEqual(S.t_start, 0.1)
        self.assertEqual(S.t_end, 200)
        self.assertEqual(S.t_period, 20)


    def test_next(self):

        S = Schedule(
            t_start=0, 
            t_period=20 
            )

        self.assertEqual(S._next(), 0)

        S.resolve(0)

        self.assertEqual(S._next(), 20)


    def test_estimate(self):

        S = Schedule(
            t_start=2, 
            t_period=20 
            )

        self.assertEqual(S.estimate(0), 2)
        self.assertEqual(S.estimate(1), 1)

        S.resolve(2)

        self.assertEqual(S.estimate(2), 20)
        self.assertEqual(S.estimate(13), 9)


    def test_detect(self):

        S = Schedule(
            t_start=2, 
            t_period=20 
            )

        S.buffer(0)

        d, c, r = S.detect(0)

        self.assertFalse(d)
        self.assertFalse(c)

        d, c, r = S.detect(4)

        self.assertTrue(d)
        self.assertFalse(c)
        self.assertEqual(r, 0.5)


    def test_detect_exact_hit_is_at_end_of_step(self):

        #the simulation calls detect(t) with t at the END of the step and
        #resolves at 'self.time + ratio * dt', so an event that sits exactly
        #on t is at ratio 1, not 0 (which would place it a full dt early)

        S = Schedule(
            t_start=2,
            t_period=20
            )

        S.buffer(1)

        d, c, r = S.detect(2)

        self.assertTrue(d)
        self.assertTrue(c)
        self.assertEqual(r, 1.0)


    def test_resolves_at_the_scheduled_time(self):

        #fixed timestep, schedule not aligned to the step grid

        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        fired = []
        S = Schedule(t_start=0.05, t_period=0.1, func_act=fired.append)

        src, sco = Constant(1.0), Scope()
        sim = Simulation(
            [src, sco],
            [Connection(src, sco)],
            events=[S],
            dt=0.01,
            log=False
            )
        sim.run(0.35, reset=True)

        self.assertEqual(len(fired), 4)
        for got, want in zip(fired, [0.05, 0.15, 0.25, 0.35]):
            self.assertAlmostEqual(got, want, places=12)


    def test_resolve_records_the_exact_scheduled_time(self):

        #the caller passes the numerically reached time which carries the
        #drift of the simulation clock, but the schedule knows its own exact
        #event time and records that instead

        fired = []
        S = Schedule(t_start=0, t_period=0.01, func_act=fired.append)

        S.resolve(0.0)
        S.resolve(0.010000000000000002)   #one ulp of drift
        S.resolve(0.020000000000000004)

        self.assertEqual(S._times, [0.0, 0.01, 0.02])
        self.assertEqual(fired, [0.0, 0.01, 0.02])


    def test_detect_absorbs_clock_drift_at_step_boundary(self):

        #after many additions of dt the simulation clock drifts away from
        #the exact schedule times; a tick that lands a few ulp behind the
        #end of the step must still be detected in that step, otherwise it
        #slips into the next one and the final tick of a run is lost

        S = Schedule(t_start=0, t_period=0.01)
        S._times = [0.0] * 10   #10 ticks resolved, next is t=0.1

        #step end one ulp below the scheduled tick at t=0.1
        t_drifted = np.nextafter(0.1, 0.0)
        self.assertLess(t_drifted, 0.1)

        S.buffer(t_drifted - 0.01)
        d, c, r = S.detect(t_drifted)

        self.assertTrue(d)
        self.assertTrue(c)
        self.assertEqual(r, 1.0)


    def test_long_run_ticks_are_complete_and_on_grid(self):

        #integration test for the drift regression: over thousands of steps
        #every tick must fire exactly once, stamped exactly on the schedule,
        #including the final tick at the end of the run

        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        fired = []
        S = Schedule(t_start=0, t_period=0.01, func_act=fired.append)

        src, sco = Constant(1.0), Scope()
        sim = Simulation(
            [src, sco],
            [Connection(src, sco)],
            events=[S],
            dt=0.01,
            log=False
            )
        sim.run(30.0, reset=True)

        #every recorded time is an exact multiple of the period
        ks = np.round(np.array(fired) / 0.01)
        self.assertEqual(np.max(np.abs(np.array(fired) - ks * 0.01)), 0.0)

        #no tick lost, none doubled, final tick at t=30 included
        self.assertTrue(np.array_equal(ks, np.arange(len(fired))))
        self.assertIn(30.0, fired)


    def test_effective_tolerance_is_capped_by_the_period(self):

        #the drift allowance must stay far below the schedule spacing so
        #no genuine tick can be absorbed by the widened tolerance

        S = Schedule(t_start=0, t_period=1e-12)

        self.assertLessEqual(S._tolerance_at(1e6), 0.1e-12)


class TestScheduleList(unittest.TestCase):
    """
    Test the implementation of the 'ScheduleList' event class.
    """

    def test_init(self):


        with self.assertRaises(ValueError):
            S = ScheduleList(times_evt=[1, 3, 5, 2, 7])

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S.times_evt, [1, 3, 5, 7])


    def test_next(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S._next(), 1)

        S.resolve(1)

        self.assertEqual(S._next(), 3)

        S.resolve(3)

        self.assertEqual(S._next(), 5)


    def test_estimate(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S.estimate(0), 1)
        self.assertEqual(S.estimate(0.5), 0.5)

        S.resolve(1)

        self.assertEqual(S.estimate(1), 2)
        self.assertEqual(S.estimate(2), 1)


    def test_detect(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        S.buffer(0)

        d, c, r = S.detect(0)

        self.assertFalse(d)
        self.assertFalse(c)

        d, c, r = S.detect(2)

        self.assertTrue(d)
        self.assertFalse(c)
        self.assertEqual(r, 0.5)


    def test_detect_exact_hit_is_at_end_of_step(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        S.buffer(0.5)

        d, c, r = S.detect(1)

        self.assertTrue(d)
        self.assertTrue(c)
        self.assertEqual(r, 1.0)


    def test_resolves_at_the_scheduled_times(self):

        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        fired = []
        S = ScheduleList(times_evt=[0.07, 0.13, 0.29], func_act=fired.append)

        src, sco = Constant(1.0), Scope()
        sim = Simulation(
            [src, sco],
            [Connection(src, sco)],
            events=[S],
            dt=0.01,
            log=False
            )
        sim.run(0.5, reset=True)

        self.assertEqual(len(fired), 3)
        for got, want in zip(fired, [0.07, 0.13, 0.29]):
            self.assertAlmostEqual(got, want, places=12)


    def test_func_act_is_not_none(self):
        def func_act(_):
            pass

        event = ScheduleList(
            times_evt=[1, 2, 3], func_act=func_act
        )

        assert event.func_act is not None

        

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
