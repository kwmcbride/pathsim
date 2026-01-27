########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk32.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK32(DiagonallyImplicitRungeKutta):
    """Four-stage, 3rd order ESDIRK method with embedded 2nd order error
    estimate. L-stable and stiffly accurate.

    Characteristics
    ---------------
    * Order: 3 (propagating) / 2 (embedded)
    * Stages: 4 (1 explicit, 3 implicit)
    * Adaptive timestep
    * L-stable, stiffly accurate
    * Stage order 2 (:math:`\\gamma = 1/2`)

    Note
    ----
    The cheapest adaptive implicit Runge-Kutta solver in this library,
    yet remarkably robust.  L-stability and stiff accuracy guarantee that
    high-frequency parasitic modes are fully damped regardless of
    timestep, and the optimal stage order of 2 (from :math:`\\gamma = 1/2`)
    minimises order reduction on stiff problems.  Three implicit stages
    per step keeps the cost well below ``ESDIRK43`` while still providing
    adaptive step-size control.  For even lower per-step cost the
    ``GEAR`` multistep solvers require only one implicit solve per step.
    Also used internally as the startup method for ``GEAR`` solvers.

    References
    ----------
    .. [1] Kennedy, C. A., & Carpenter, M. H. (2019). "Diagonally implicit
           Runge-Kutta methods for stiff ODEs". Applied Numerical
           Mathematics, 146, 221-244.
           :doi:`10.1016/j.apnum.2019.07.008`
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme and embedded method
        self.n = 3
        self.m = 2

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0, 3/2, 1.0]

        #butcher table
        self.BT = {
            0: None,      #explicit first stage
            1: [1/2, 1/2],
            2: [5/8, 3/8, 1/2],
            3: [7/18, 1/3, -2/9, 1/2]
            }

        #coefficients for truncation error estimate
        self.TR = [-1/9, -1/6, -2/9, 1/2]