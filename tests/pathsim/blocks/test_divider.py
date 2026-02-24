########################################################################################
##
##                                  TESTS FOR
##                             'blocks.divider.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.divider import Divider

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestDivider(unittest.TestCase):
    """
    Test the implementation of the 'Divider' block class
    """

    def test_init(self):

        # default initialization
        D = Divider()
        self.assertIsNone(D.operations)

        # valid ops strings
        for ops in ["*", "/", "*/", "/*", "**/", "/**"]:
            D = Divider(ops)
            self.assertEqual(D.operations, ops)

        # non-string types are rejected
        for bad in [0.4, 3, [1, -1], True]:
            with self.assertRaises(ValueError):
                Divider(bad)

        # strings with invalid characters are rejected
        for bad in ["+/", "*-", "a", "**0", "+-"]:
            with self.assertRaises(ValueError):
                Divider(bad)


    def test_embedding(self):
        """Test algebraic output against reference via Embedding."""

        # default: multiply all (identical to Multiplier)
        D = Divider()

        def src(t): return np.cos(t), np.sin(t) + 2, 3.0, t + 1
        def ref(t): return np.cos(t) * (np.sin(t) + 2) * 3.0 * (t + 1)

        E = Embedding(D, src, ref)
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # '**/' : multiply first two, divide by third
        D = Divider('**/')

        def src(t): return np.cos(t) + 2, np.sin(t) + 2, t + 1
        def ref(t): return (np.cos(t) + 2) * (np.sin(t) + 2) / (t + 1)

        E = Embedding(D, src, ref)
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # '*/' : u0 / u1
        D = Divider('*/')

        def src(t): return t + 1, np.cos(t) + 2
        def ref(t): return (t + 1) / (np.cos(t) + 2)

        E = Embedding(D, src, ref)
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # ops string shorter than number of inputs: extra inputs default to '*'
        # '/' with 3 inputs → y = u1 * u2 / u0
        D = Divider('/')

        def src(t): return t + 1, np.cos(t) + 2, 3.0
        def ref(t): return (np.cos(t) + 2) * 3.0 / (t + 1)

        E = Embedding(D, src, ref)
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # single input, default: passes through unchanged
        D = Divider()

        def src(t): return np.cos(t)
        def ref(t): return np.cos(t)

        E = Embedding(D, src, ref)
        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_linearization(self):
        """Test linearize / delinearize round-trip."""

        # default (multiply all) — nonlinear, so only check at linearization point
        D = Divider()

        def src(t): return np.cos(t) + 2, t + 1, 3.0
        def ref(t): return (np.cos(t) + 2) * (t + 1) * 3.0

        E = Embedding(D, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # linearize at the current operating point (inputs set to src(9) by the loop)
        D.linearize(t)
        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, 8)

        D.delinearize()
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        # with ops string
        D = Divider('**/')

        def src(t): return np.cos(t) + 2, np.sin(t) + 2, t + 1
        def ref(t): return (np.cos(t) + 2) * (np.sin(t) + 2) / (t + 1)

        E = Embedding(D, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        D.linearize(t)
        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, 8)

        D.delinearize()
        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_update_single(self):

        D = Divider()

        D.inputs[0] = 5.0
        D.update(None)

        self.assertEqual(D.outputs[0], 5.0)


    def test_update_multi(self):

        D = Divider()

        D.inputs[0] = 2.0
        D.inputs[1] = 3.0
        D.inputs[2] = 4.0
        D.update(None)

        self.assertEqual(D.outputs[0], 24.0)


    def test_update_ops(self):

        # '**/' : 2 * 3 / 4 = 1.5
        D = Divider('**/')
        D.inputs[0] = 2.0
        D.inputs[1] = 3.0
        D.inputs[2] = 4.0
        D.update(None)
        self.assertAlmostEqual(D.outputs[0], 1.5)

        # '*/' : 6 / 2 = 3
        D = Divider('*/')
        D.inputs[0] = 6.0
        D.inputs[1] = 2.0
        D.update(None)
        self.assertAlmostEqual(D.outputs[0], 3.0)

        # '/' with extra inputs: 2 * 3 / 4 = 1.5  (u0 divides, u1 u2 multiply)
        D = Divider('/')
        D.inputs[0] = 4.0
        D.inputs[1] = 2.0
        D.inputs[2] = 3.0
        D.update(None)
        self.assertAlmostEqual(D.outputs[0], 1.5)

        # '/**' : u1 * u2 / u0
        D = Divider('/**')
        D.inputs[0] = 4.0
        D.inputs[1] = 2.0
        D.inputs[2] = 3.0
        D.update(None)
        self.assertAlmostEqual(D.outputs[0], 1.5)


    def test_jacobian(self):
        """Verify analytical Jacobian against central finite differences."""

        eps = 1e-6

        def numerical_jac(func, u):
            n = len(u)
            J = np.zeros((1, n))
            for k in range(n):
                u_p = u.copy(); u_p[k] += eps
                u_m = u.copy(); u_m[k] -= eps
                J[0, k] = (func(u_p) - func(u_m)) / (2 * eps)
            return J

        # default (all multiply)
        D = Divider()
        u = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            D.op_alg.jac(u),
            numerical_jac(D.op_alg._func, u),
            rtol=1e-5,
        )

        # '**/' : u0 * u1 / u2
        D = Divider('**/')
        u = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            D.op_alg.jac(u),
            numerical_jac(D.op_alg._func, u),
            rtol=1e-5,
        )

        # '*/' : u0 / u1
        D = Divider('*/')
        u = np.array([6.0, 2.0])
        np.testing.assert_allclose(
            D.op_alg.jac(u),
            numerical_jac(D.op_alg._func, u),
            rtol=1e-5,
        )

        # '/**' : u1 * u2 / u0
        D = Divider('/**')
        u = np.array([4.0, 2.0, 3.0])
        np.testing.assert_allclose(
            D.op_alg.jac(u),
            numerical_jac(D.op_alg._func, u),
            rtol=1e-5,
        )

        # ops shorter than inputs: '/' with 3 inputs → u1 * u2 / u0
        D = Divider('/')
        u = np.array([4.0, 2.0, 3.0])
        np.testing.assert_allclose(
            D.op_alg.jac(u),
            numerical_jac(D.op_alg._func, u),
            rtol=1e-5,
        )


    def test_zero_div(self):

        # 'warn' (default): produces inf, no exception
        D = Divider('*/', zero_div='warn')
        D.inputs[0] = 6.0
        D.inputs[1] = 0.0
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            D.update(None)
        self.assertTrue(np.isinf(D.outputs[0]))

        # 'raise': ZeroDivisionError on zero denominator
        D = Divider('*/', zero_div='raise')
        D.inputs[0] = 6.0
        D.inputs[1] = 0.0
        with self.assertRaises(ZeroDivisionError):
            D.update(None)

        # 'raise': no error when denominator is nonzero
        D = Divider('*/', zero_div='raise')
        D.inputs[0] = 6.0
        D.inputs[1] = 2.0
        D.update(None)
        self.assertAlmostEqual(D.outputs[0], 3.0)

        # 'clamp': output is large-but-finite
        D = Divider('*/', zero_div='clamp')
        D.inputs[0] = 1.0
        D.inputs[1] = 0.0
        D.update(None)
        self.assertTrue(np.isfinite(D.outputs[0]))
        self.assertGreater(abs(D.outputs[0]), 1.0)

        # 'raise' invalid zero_div value
        with self.assertRaises(ValueError):
            Divider('*/', zero_div='ignore')

        # Jacobian: 'raise' on zero denominator input
        D = Divider('*/', zero_div='raise')
        with self.assertRaises(ZeroDivisionError):
            D.op_alg.jac(np.array([6.0, 0.0]))

        # Jacobian: 'clamp' stays finite
        D = Divider('*/', zero_div='clamp')
        J = D.op_alg.jac(np.array([1.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(J)))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
