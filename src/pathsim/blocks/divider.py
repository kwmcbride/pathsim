#########################################################################################
##
##                        REDUCTION BLOCKS (blocks/divider.py)
##
##                    This module defines static 'Divider' block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from math import prod

from ._block import Block
from ..utils.register import Register
from ..optim.operator import Operator
from ..utils.mutable import mutable


# MISO BLOCKS ===========================================================================

_ZERO_DIV_OPTIONS = ("warn", "raise", "clamp")


@mutable
class Divider(Block):
    """Multiplies and divides input signals (MISO).

    This is the default behavior (multiply all):

    .. math::

        y(t) = \\prod_i u_i(t)

    and this is the behavior with an operations string:

    .. math::

        y(t) = \\frac{\\prod_{i \\in M} u_i(t)}{\\prod_{j \\in D} u_j(t)}

    where :math:`M` is the set of inputs with ``*`` and :math:`D` the set with ``/``.


    Example
    -------
    Default initialization multiplies the first input and divides by the second:

    .. code-block:: python

        D = Divider()

    Multiply the first two inputs and divide by the third:

    .. code-block:: python

        D = Divider('**/')

    Raise an error instead of producing ``inf`` when a denominator input is zero:

    .. code-block:: python

        D = Divider('**/', zero_div='raise')

    Clamp the denominator to machine epsilon so the output stays finite:

    .. code-block:: python

        D = Divider('**/', zero_div='clamp')


    Note
    ----
    This block is purely algebraic and its operation (``op_alg``) will be called
    multiple times per timestep, each time when ``Simulation._update(t)`` is
    called in the global simulation loop.


    Parameters
    ----------
    operations : str, optional
        String of ``*`` and ``/`` characters indicating which inputs are
        multiplied (``*``) or divided (``/``). Inputs beyond the length of
        the string default to ``*``. Defaults to ``'*/'`` (divide second
        input by first).
    zero_div : str, optional
        Behaviour when a denominator input is zero. One of:

        ``'warn'`` *(default)*
            Propagates ``inf`` and emits a ``RuntimeWarning`` — numpy's
            standard behaviour.
        ``'raise'``
            Raises ``ZeroDivisionError``.
        ``'clamp'``
            Clamps the denominator magnitude to machine epsilon
            (``numpy.finfo(float).eps``), preserving sign, so the output
            stays large-but-finite rather than ``inf``.


    Attributes
    ----------
    _ops : dict
        Maps operation characters to exponent values (``+1`` or ``-1``).
    _ops_array : numpy.ndarray
        Exponents (+1 for ``*``, -1 for ``/``) converted to an array.
    op_alg : Operator
        Internal algebraic operator.
    """

    input_port_labels = None
    output_port_labels = {"out": 0}

    def __init__(self, operations="*/", zero_div="warn"):
        super().__init__()

        # validate zero_div
        if zero_div not in _ZERO_DIV_OPTIONS:
            raise ValueError(
                f"'zero_div' must be one of {_ZERO_DIV_OPTIONS}, got '{zero_div}'"
            )
        self.zero_div = zero_div

        # allowed arithmetic operations mapped to exponents
        self._ops = {"*": 1, "/": -1}
        self.operations = operations

        if self.operations is None:

            # Default: multiply all inputs — identical to Multiplier
            self.op_alg = Operator(
                func=prod,
                jac=lambda x: np.array([[
                    prod(np.delete(x, i)) for i in range(len(x))
                ]])
            )

        else:

            # input validation
            if not isinstance(self.operations, str):
                raise ValueError("'operations' must be a string or None")
            for op in self.operations:
                if op not in self._ops:
                    raise ValueError(
                        f"operation '{op}' not in {set(self._ops)}"
                    )

            self._ops_array = np.array(
                [self._ops[op] for op in self.operations], dtype=float
            )

            # capture for closures
            _ops_array = self._ops_array
            _zero_div = zero_div
            _eps = np.finfo(float).eps

            def _safe_den(d):
                """Apply zero_div policy to a denominator value."""
                if d == 0:
                    if _zero_div == "raise":
                        raise ZeroDivisionError(
                            "Divider: denominator is zero. "
                            "Use zero_div='warn' or 'clamp' to suppress."
                        )
                    elif _zero_div == "clamp":
                        return _eps
                return d

            def prod_ops(X):
                n = len(X)
                no = len(_ops_array)
                ops = np.ones(n)
                ops[:min(n, no)] = _ops_array[:min(n, no)]
                num = prod(X[i] for i in range(n) if ops[i] > 0)
                den = _safe_den(prod(X[i] for i in range(n) if ops[i] < 0))
                return num / den

            def jac_ops(X):
                n = len(X)
                no = len(_ops_array)
                ops = np.ones(n)
                ops[:min(n, no)] = _ops_array[:min(n, no)]
                X = np.asarray(X, dtype=float)
                # Apply zero_div policy to all denominator inputs up front so
                # both the direct division and the rest-product stay consistent.
                X_safe = X.copy()
                for i in range(n):
                    if ops[i] < 0:
                        X_safe[i] = _safe_den(float(X[i]))
                row = []
                for k in range(n):
                    rest = np.prod(
                        np.power(np.delete(X_safe, k), np.delete(ops, k))
                    )
                    if ops[k] > 0:  # multiply: dy/du_k = prod of rest
                        row.append(rest)
                    else:           # divide:   dy/du_k = -rest / u_k^2
                        row.append(-rest / X_safe[k] ** 2)
                return np.array([row])

            self.op_alg = Operator(func=prod_ops, jac=jac_ops)


    def __len__(self):
        """Purely algebraic block."""
        return 1


    def update(self, t):
        """Update system equation.

        Parameters
        ----------
        t : float
            Evaluation time.
        """
        u = self.inputs.to_array()
        self.outputs.update_from_array(self.op_alg(u))
