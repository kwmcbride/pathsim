#########################################################################################
##
##                             STANDARD INTEGRATOR BLOCK 
##                              (blocks/integrator.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import DynamicOperator


# BLOCKS ================================================================================

class Integrator(Block):
    """Integrates the input signal.

    Uses a numerical integration engine like this:

    .. math::

        y(t) = \\int_0^t u(\\tau) \\ d \\tau
    
    or in differential form like this:

    .. math::
        \\begin{align}
            \\dot{x}(t) &= u(t) \\\\
                   y(t) &= x(t)
        \\end{align}

    The Integrator block is inherently MIMO capable, so `u` 
    and `y` can be vectors.
    
    Example
    -------
    This is how to initialize the integrator: 

    .. code-block:: python
    
        #initial value 0.0
        i1 = Integrator()

        #initial value 2.5
        i2 = Integrator(2.5)
    

    Parameters
    ----------
    initial_value : float, array
        initial value of integrator
    """

    def __init__(self, initial_value=0.0):
        super().__init__()

        #save initial value
        self.initial_value = initial_value


    def __len__(self):
        return 0


    def update(self, t):
        """update system equation fixed point loop

        Note
        ----
        integrator does not have passthrough, therefore this 
        method is performance optimized for this case

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs.update_from_array(self.engine.state)


    def to_statespace(self, t):
        """Local linear state space model of the integrator.

        The integrator is already linear, so its model is exact and
        independent of the operating point

        .. math::

            \\mathbf{A} = 0, \\quad \\mathbf{B} = \\mathbf{I}, \\quad
            \\mathbf{C} = \\mathbf{I}, \\quad \\mathbf{D} = 0


        Note
        ----
        The integrator carries no operators, so the generic implementation
        of 'Block.to_statespace' cannot derive this and would reject it.

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        A, B, C, D : np.ndarray
            local state space matrices of the integrator
        """
        #no engine assigned yet -> no state, same degradation as the base class
        nx = len(np.atleast_1d(self.engine.state)) if self.engine else 0
        nu = len(self.inputs.to_array())
        ny = len(self.outputs.to_array())

        return (
            np.zeros((nx, nx)),
            np.eye(nx, nu),
            np.eye(ny, nx),
            np.zeros((ny, nu))
            )


    def derivative(self, t):
        """time derivative of the integrator state, which is just the input

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        dxdt : np.ndarray
            time derivative of the integrator state
        """
        return self.inputs.to_array()


    def solve(self, t, dt):
        """advance solution of implicit update equation of the solver

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        -------
        error : float
            solver residual norm
        """
        f = self.inputs.to_array()
        return self.engine.solve(f, None, dt)


    def step(self, t, dt):
        """compute timestep update with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        f = self.inputs.to_array()
        return self.engine.step(f, dt)