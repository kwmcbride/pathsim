########################################################################################
##
##                                  TESTS FOR
##                            'optim/linearization.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Constant, Adder, Amplifier, Integrator, StateSpace, Scope, Comparator
    )
from pathsim.exceptions import LinearizationError


# TESTS ================================================================================

class TestBlockToStateSpace(unittest.TestCase):
    """
    Test the local linear models the blocks return from 'to_statespace'
    """

    def test_algebraic_block(self):
        """purely algebraic blocks only have a feedthrough matrix"""

        amp = Amplifier(3.0)
        amp.inputs.update_from_array([1.0])
        amp.update(0.0)

        A, B, C, D = amp.to_statespace(0.0)

        self.assertEqual(A.shape, (0, 0))
        self.assertEqual(B.shape, (0, 1))
        self.assertEqual(C.shape, (1, 0))
        self.assertTrue(np.allclose(D, [[3.0]]))


    def test_statespace_block_is_exact(self):
        """the linear model of a 'StateSpace' block is the block itself"""

        A = np.array([[-1.0, 0.5], [0.0, -2.0]])
        B = np.array([[1.0], [2.0]])
        C = np.array([[1.0, 1.0]])
        D = np.array([[0.3]])

        blk = StateSpace(A=A, B=B, C=C, D=D)
        Sim = Simulation(blocks=[blk], connections=[], log=False)
        Sim._update(0.0)

        _A, _B, _C, _D = blk.to_statespace(0.0)

        self.assertTrue(np.allclose(_A, A))
        self.assertTrue(np.allclose(_B, B))
        self.assertTrue(np.allclose(_C, C))
        self.assertTrue(np.allclose(_D, D))


    def test_integrator_is_exact(self):
        """the integrator model is exact and independent of the operating point"""

        integ = Integrator(0.0)
        Sim = Simulation(blocks=[integ], connections=[], log=False)
        Sim._update(0.0)

        A, B, C, D = integ.to_statespace(0.0)

        self.assertTrue(np.allclose(A, [[0.0]]))
        self.assertTrue(np.allclose(B, [[1.0]]))
        self.assertTrue(np.allclose(C, [[1.0]]))
        self.assertTrue(np.allclose(D, [[0.0]]))


    def test_to_statespace_is_a_pure_query(self):
        """asking for the model must not switch the block to its surrogate"""

        from pathsim.blocks import DynamicalSystem

        def build():
            plant = DynamicalSystem(
                func_dyn=lambda x, u, t: -x**2 + u,
                func_alg=lambda x, u, t: x,
                initial_value=1.0,
                jac_dyn=lambda x, u, t: -2*x
                )
            src, sco = Constant(4.0), Scope()
            return Simulation(
                blocks=[src, plant, sco],
                connections=[Connection(src, plant), Connection(plant, sco)],
                dt=0.01, log=False
                ), plant, sco

        #reference run of the untouched nonlinear system
        Sim_ref, _, Sco_ref = build()
        Sim_ref.run(duration=2.0)
        _, (y_ref,) = Sco_ref.read()

        #same run, but the model is queried first
        Sim, plant, Sco = build()
        Sim._update(0.0)
        plant.to_statespace(0.0)
        Sim.run(duration=2.0)
        _, (y,) = Sco.read()

        self.assertTrue(np.allclose(y, y_ref))


    def test_linearize_returns_the_model(self):
        """'linearize' switches the block over and hands back the same model"""

        amp = Amplifier(3.0)
        amp.inputs.update_from_array([1.0])
        amp.update(0.0)

        query = amp.to_statespace(0.0)
        switch = amp.linearize(0.0)

        for _q, _s in zip(query, switch):
            self.assertTrue(np.allclose(_q, _s))

        amp.delinearize()


    def test_non_linearizable_block_raises(self):
        """blocks without a valid linear model fail loudly"""

        with self.assertRaises(LinearizationError):
            Comparator().to_statespace(0.0)


    def test_non_linearizable_block_raises_from_simulation(self):
        """and the failure propagates to the system level"""

        Sim = Simulation(blocks=[Comparator()], connections=[], log=False)

        with self.assertRaises(LinearizationError):
            Sim.linearize()

        with self.assertRaises(LinearizationError):
            Sim.to_statespace(inputs=[], outputs=[])


class TestSimulationToStateSpace(unittest.TestCase):
    """
    Test the global state space model assembled by 'Simulation.to_statespace'
    """

    def test_cascade(self):
        """u -> gain(3) -> int -> gain(2) -> int, checked against the analytic model"""

        src = Constant(0.0)
        g1, g2 = Amplifier(3.0), Amplifier(2.0)
        i1, i2 = Integrator(0.0), Integrator(0.0)

        Sim = Simulation(
            blocks=[src, g1, i1, g2, i2],
            connections=[
                Connection(src, g1),
                Connection(g1, i1),
                Connection(i1, g2),
                Connection(g2, i2)
                ],
            log=False
            )

        ss = Sim.to_statespace(inputs=[g1[0]], outputs=[i2[0]])

        self.assertTrue(np.allclose(ss.A, [[0, 0], [2, 0]]))
        self.assertTrue(np.allclose(ss.B, [[3], [0]]))
        self.assertTrue(np.allclose(ss.C, [[0, 1]]))
        self.assertTrue(np.allclose(ss.D, [[0]]))


    def test_labels_are_consistent_across_categories(self):
        """the same block is named the same in every label list"""

        src = Constant(0.0)
        g1, g2 = Amplifier(3.0), Amplifier(2.0)
        i1, i2 = Integrator(0.0), Integrator(0.0)

        Sim = Simulation(
            blocks=[src, g1, i1, g2, i2],
            connections=[
                Connection(src, g1),
                Connection(g1, i1),
                Connection(i1, g2),
                Connection(g2, i2)
                ],
            log=False
            )

        ss = Sim.to_statespace(inputs=[g2[0]], outputs=[i2[0]])

        #'i2' is the second integrator, so it must not be called 'Integrator_0'
        self.assertEqual(ss.state_labels, ["Integrator_0", "Integrator_1"])
        self.assertEqual(ss.output_labels, ["Integrator_1"])
        self.assertEqual(ss.input_labels, ["Amplifier_1"])


    def test_negative_feedback(self):
        """closing a loop with gain 'k' around an integrator gives 'A = -k'"""

        k = 5.0
        ref, add = Constant(0.0), Adder("+-")
        integ, gain = Integrator(0.0), Amplifier(k)

        Sim = Simulation(
            blocks=[ref, add, integ, gain],
            connections=[
                Connection(ref, add[0]),
                Connection(integ, gain),
                Connection(gain, add[1]),
                Connection(add, integ)
                ],
            log=False
            )

        ss = Sim.to_statespace(inputs=[add[0]], outputs=[integ[0]])

        self.assertTrue(np.allclose(ss.A, [[-k]]))
        self.assertTrue(np.allclose(ss.B, [[1.0]]))
        self.assertTrue(np.allclose(ss.C, [[1.0]]))
        self.assertTrue(np.allclose(ss.D, [[0.0]]))


    def test_mimo_roundtrip(self):
        """a lone MIMO 'StateSpace' block assembles back to its own matrices"""

        A = np.array([[-1.0, 0.5], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 2.0]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.array([[0.0, 0.3], [0.1, 0.0]])

        blk = StateSpace(A=A, B=B, C=C, D=D)
        Sim = Simulation(blocks=[blk], connections=[], log=False)

        ss = Sim.to_statespace(inputs=[blk[0, 1]], outputs=[blk[0, 1]])

        self.assertTrue(np.allclose(ss.A, A))
        self.assertTrue(np.allclose(ss.B, B))
        self.assertTrue(np.allclose(ss.C, C))
        self.assertTrue(np.allclose(ss.D, D))


    def test_algebraic_loop_is_resolved(self):
        """an algebraic loop surviving the break is eliminated, not rejected

        Two gains of 0.5 in a loop fed through an adder give a closed loop
        gain of 0.5 / (1 - 0.25).
        """

        src, add = Constant(0.0), Adder("++")
        a, b = Amplifier(0.5), Amplifier(0.5)

        Sim = Simulation(
            blocks=[src, add, a, b],
            connections=[
                Connection(src, add[0]),
                Connection(add, a),
                Connection(a, b),
                Connection(b, add[1])
                ],
            log=False
            )

        ss = Sim.to_statespace(inputs=[add[0]], outputs=[a[0]])

        self.assertTrue(np.allclose(ss.D, [[0.5 / (1 - 0.25)]]))


    def test_ill_posed_loop_raises(self):
        """a unity gain algebraic loop has no linear model and must fail"""

        a, b = Amplifier(1.0), Amplifier(1.0)

        Sim = Simulation(
            blocks=[a, b],
            connections=[Connection(a, b), Connection(b, a)],
            log=False
            )

        with self.assertRaises(LinearizationError):
            Sim.to_statespace(inputs=[], outputs=[a[0]])


    def test_to_statespace_is_a_pure_query(self):
        """assembling the model must leave the system nonlinear"""

        from pathsim.blocks import DynamicalSystem

        def build():
            plant = DynamicalSystem(
                func_dyn=lambda x, u, t: -x**2 + u,
                func_alg=lambda x, u, t: x,
                initial_value=1.0,
                jac_dyn=lambda x, u, t: -2*x
                )
            src, sco = Constant(4.0), Scope()
            return Simulation(
                blocks=[src, plant, sco],
                connections=[Connection(src, plant), Connection(plant, sco)],
                dt=0.01, log=False
                ), plant, sco

        Sim_ref, _, Sco_ref = build()
        Sim_ref.run(duration=2.0)
        _, (y_ref,) = Sco_ref.read()

        Sim, plant, Sco = build()
        Sim.to_statespace(inputs=[plant[0]], outputs=[plant[0]])
        Sim.run(duration=2.0)
        _, (y,) = Sco.read()

        self.assertTrue(np.allclose(y, y_ref))


    def test_small_signal_response_matches_the_nonlinear_system(self):
        """the assembled model reproduces the small signal step response

        A nonlinear plant in a closed loop is linearized in its operating
        point, then driven by the same small reference step as the original
        system. The deviation from the operating point has to agree.
        """

        from pathsim.blocks import DynamicalSystem
        from pathsim.blocks.ctrl import PID

        ref, err = Constant(1.0), Adder("+-")
        ctrl = PID(Kp=1.0, Ki=2.0, Kd=0.0, f_max=50)
        plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2*x
            )
        Sco = Scope()

        Sim = Simulation(
            blocks=[ref, err, ctrl, plant, Sco],
            connections=[
                Connection(ref, err[0]),
                Connection(plant, err[1]),
                Connection(err, ctrl),
                Connection(ctrl, plant),
                Connection(plant, Sco)
                ],
            dt=0.01, log=False
            )

        #drive the system into its operating point, the integral action of
        #the controller settles the output at the reference
        Sim.run(duration=20.0)
        y0 = plant.outputs[0]
        self.assertAlmostEqual(y0, 1.0, 4)

        #model around that point, breaking the loop at the reference input
        ss = Sim.to_statespace(inputs=[err[0]], outputs=[plant[0]])

        self.assertEqual(ss.A.shape, (3, 3))
        self.assertEqual(ss.B.shape, (3, 1))
        self.assertEqual(ss.C.shape, (1, 3))
        self.assertEqual(ss.D.shape, (1, 1))

        #small reference step on the original nonlinear system, recording
        #only the response to the step itself
        delta = 0.01
        Sco.reset()
        ref.value += delta
        Sim.run(duration=3.0, reset=False)
        t_full, (y_full,) = Sco.read()

        #same step on the assembled model alone
        step, Sco2 = Constant(delta), Scope()
        Sim2 = Simulation(
            blocks=[step, ss, Sco2],
            connections=[Connection(step, ss), Connection(ss, Sco2)],
            dt=0.01, log=False
            )
        Sim2.run(duration=3.0, reset=False)
        t_lin, (y_lin,) = Sco2.read()

        #both runs are adaptive, so compare on a common time grid
        y_lin = np.interp(t_full - t_full[0], t_lin - t_lin[0], y_lin)

        self.assertTrue(np.max(np.abs((np.asarray(y_full) - y0) - y_lin)) < 1e-3)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
