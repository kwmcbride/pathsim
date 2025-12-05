########################################################################################
##
##                      Testing System with Model Exchange FMU
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os

from pathlib import Path

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR.parent.parent / "docs" / "source" / "examples" / "data"

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ModelExchangeFMU
from pathsim.solvers import RKBS32, RKDP54


# TESTCASES ============================================================================

@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestModelExchangeFMUBouncingBall(unittest.TestCase):
    """Test Model Exchange FMU integration with the Bouncing Ball example"""

    @classmethod
    def setUpClass(cls):
        """Check if test FMU exists"""
        cls.fmu_path = TEST_DATA_DIR / "BouncingBall_ME.fmu"
        if not cls.fmu_path.exists():
            raise unittest.SkipTest(f"Test FMU not found: {cls.fmu_path}")

    def setUp(self):
        """Set up the bouncing ball system"""
        self.fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="bouncing_ball",
            start_values={"e": 0.7},
            tolerance=1e-10,
            verbose=False
        )
        self.sco = Scope(labels=["h", "v"])

        self.sim = Simulation(
            blocks=[self.fmu, self.sco],
            connections=[
                Connection(self.fmu[0], self.sco[0]),  # height
                Connection(self.fmu[1], self.sco[1]),  # velocity
            ],
            dt=0.01,
            dt_max=0.01,
            Solver=RKBS32,
            tolerance_lte_rel=1e-6,
            tolerance_lte_abs=1e-9,
            log=False
        )

    def test_fmu_wrapper_accessible(self):
        """Test that fmu_wrapper is accessible"""
        self.assertIsNotNone(self.fmu.fmu_wrapper)
        self.assertEqual(self.fmu.fmu_wrapper.mode, 'model_exchange')

    def test_fmu_states(self):
        """Test FMU has correct number of states"""
        self.assertEqual(self.fmu.fmu_wrapper.n_states, 2)

    def test_fmu_event_indicators(self):
        """Test FMU has event indicators"""
        self.assertGreaterEqual(self.fmu.fmu_wrapper.n_event_indicators, 1)

    def test_simulation_runs(self):
        """Test that simulation runs without errors"""
        result = self.sim.run(2.0)

        self.assertIn('total_steps', result)
        self.assertIn('successful_steps', result)
        self.assertGreater(result['successful_steps'], 0)

    def test_bouncing_behavior(self):
        """Test that ball bounces (height stays non-negative)"""
        self.sim.run(4.0)

        time, (h, v) = self.sco.read()

        # Height should never go significantly negative
        self.assertTrue(np.all(h >= -0.01))

        # Should have some bounces (velocity sign changes)
        sign_changes = np.sum(np.diff(np.sign(v)) != 0)
        self.assertGreater(sign_changes, 0)

    def test_events_detected(self):
        """Test that bounce events are detected"""
        self.sim.run(4.0)

        # Should have at least one zero-crossing event
        total_events = sum(len(evt) for evt in self.fmu.events)
        self.assertGreater(total_events, 0)

    def test_energy_dissipation(self):
        """Test that energy is dissipated (restitution < 1)"""
        self.sim.run(4.0)

        time, (h, v) = self.sco.read()

        # Maximum height should decrease over time
        # Find local maxima of height
        local_max_indices = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i-1] and h[i] > h[i+1]:
                local_max_indices.append(i)

        if len(local_max_indices) >= 2:
            # Later maxima should be lower
            self.assertLess(h[local_max_indices[-1]], h[local_max_indices[0]])

    def test_reset(self):
        """Test that FMU can be reset and re-run"""
        # Run once
        self.sim.run(1.0)
        time1, (h1, v1) = self.sco.read()

        # Reset
        self.sim.reset()

        # Run again
        self.sim.run(1.0)
        time2, (h2, v2) = self.sco.read()

        # Results should be similar
        np.testing.assert_array_almost_equal(h1, h2, decimal=5)
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)


@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestModelExchangeFMUVanDerPol(unittest.TestCase):
    """Test Model Exchange FMU integration with the Van der Pol oscillator"""

    @classmethod
    def setUpClass(cls):
        """Check if test FMU exists"""
        cls.fmu_path = TEST_DATA_DIR / "VanDerPol_ME.fmu"
        if not cls.fmu_path.exists():
            raise unittest.SkipTest(f"Test FMU not found: {cls.fmu_path}")

    def setUp(self):
        """Set up the Van der Pol system"""
        self.fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="vanderpol",
            start_values={
                "mu": 1.0,
                "x0": 2.0,
                "x1": 0.0,
            },
            tolerance=1e-8,
            verbose=False
        )
        self.sco = Scope(labels=["x0", "x1"])

        self.sim = Simulation(
            blocks=[self.fmu, self.sco],
            connections=[
                Connection(self.fmu[0], self.sco[0]),
                Connection(self.fmu[1], self.sco[1]),
            ],
            dt=0.1,
            dt_max=0.1,
            Solver=RKDP54,
            tolerance_lte_rel=1e-6,
            tolerance_lte_abs=1e-9,
            log=False
        )

    def test_fmu_states(self):
        """Test FMU has correct number of states"""
        self.assertEqual(self.fmu.fmu_wrapper.n_states, 2)

    def test_no_event_indicators(self):
        """Test Van der Pol has no event indicators (pure continuous)"""
        self.assertEqual(self.fmu.fmu_wrapper.n_event_indicators, 0)

    def test_simulation_runs(self):
        """Test that simulation runs without errors"""
        result = self.sim.run(10.0)

        self.assertIn('total_steps', result)
        self.assertGreater(result['successful_steps'], 0)

    def test_oscillation(self):
        """Test that the system oscillates"""
        self.sim.run(30.0)

        time, (x0, x1) = self.sco.read()

        # x0 should oscillate (have both positive and negative values)
        self.assertTrue(np.any(x0 > 0))
        self.assertTrue(np.any(x0 < 0))

        # Should have multiple zero crossings
        sign_changes = np.sum(np.diff(np.sign(x0)) != 0)
        self.assertGreater(sign_changes, 4)  # At least 2 full cycles

    def test_limit_cycle(self):
        """Test that the system approaches a limit cycle"""
        self.sim.run(50.0)

        time, (x0, x1) = self.sco.read()

        # For Van der Pol with mu=1, the limit cycle amplitude is approximately 2
        # After sufficient time, max amplitude should be close to 2
        late_x0 = x0[len(x0)//2:]  # Second half of simulation
        max_amplitude = np.max(np.abs(late_x0))

        self.assertGreater(max_amplitude, 1.5)
        self.assertLess(max_amplitude, 2.5)

    def test_reset(self):
        """Test that FMU can be reset and re-run"""
        # Run once
        self.sim.run(5.0)
        time1, (x0_1, x1_1) = self.sco.read()

        # Reset
        self.sim.reset()

        # Run again
        self.sim.run(5.0)
        time2, (x0_2, x1_2) = self.sco.read()

        # Results should be similar
        np.testing.assert_array_almost_equal(x0_1, x0_2, decimal=4)
        np.testing.assert_array_almost_equal(x1_1, x1_2, decimal=4)


@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestModelExchangeFMUBlockAPI(unittest.TestCase):
    """Test ModelExchangeFMU block API"""

    @classmethod
    def setUpClass(cls):
        """Check if test FMU exists"""
        cls.fmu_path = TEST_DATA_DIR / "BouncingBall_ME.fmu"
        if not cls.fmu_path.exists():
            raise unittest.SkipTest(f"Test FMU not found: {cls.fmu_path}")

    def test_fmu_wrapper_model_description(self):
        """Test accessing model description through fmu_wrapper"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        md = fmu.fmu_wrapper.model_description
        self.assertIsNotNone(md)
        self.assertIsNotNone(md.modelName)

    def test_fmu_wrapper_fmi_version(self):
        """Test accessing FMI version through fmu_wrapper"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        version = fmu.fmu_wrapper.fmi_version
        self.assertTrue(version.startswith('2.') or version.startswith('3.'))

    def test_fmu_wrapper_n_states(self):
        """Test accessing n_states through fmu_wrapper"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        self.assertEqual(fmu.fmu_wrapper.n_states, 2)

    def test_fmu_wrapper_n_event_indicators(self):
        """Test accessing n_event_indicators through fmu_wrapper"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        self.assertGreaterEqual(fmu.fmu_wrapper.n_event_indicators, 1)

    def test_fmu_wrapper_output_refs(self):
        """Test accessing output_refs through fmu_wrapper"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        self.assertIsInstance(fmu.fmu_wrapper.output_refs, dict)

    def test_events_list_created(self):
        """Test that events list is properly created"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10
        )

        # Should have at least one event (for event indicators)
        self.assertGreater(len(fmu.events), 0)

    def test_verbose_mode(self):
        """Test verbose mode doesn't crash"""
        fmu = ModelExchangeFMU(
            fmu_path=str(self.fmu_path),
            instance_name="test",
            tolerance=1e-10,
            verbose=True
        )

        # Just verify it initialized without error
        self.assertTrue(fmu.verbose)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
