#########################################################################################
##
##                              PATHSIM EXCEPTIONS
##                               (exceptions.py)
##
##      This module defines custom exceptions for the PathSim simulation framework.
##
#########################################################################################


class StopSimulation(Exception):
    """Exception that can be raised by blocks or models to signal that the 
    simulation should stop immediately.

    This provides a clean mechanism for user-defined stopping conditions,
    such as reaching a target state, detecting a fault, or satisfying 
    a convergence criterion.

    When raised inside a block's update, sample, or any other method 
    called during the simulation loop, the 'Simulation' class will catch 
    it gracefully and terminate the run as if 'stop()' had been called.

    Parameters
    ----------
    message : str
        optional message describing the stopping condition

    Example
    -------

    Raise from inside a block to stop the simulation:

    .. code-block:: python

        from pathsim.exceptions import StopSimulation
        from pathsim.blocks import Function

        def check(x):
            if x > 10.0:
                raise StopSimulation(f"value exceeded threshold: {x:.4f}")
            return x

        blk = Function(check)
    """


class LinearizationError(Exception):
    """Exception that is raised when a block has no valid linear state space
    model in the current operating point.

    Not every block can be linearized. Blocks with a switching or discontinuous
    characteristic ('Comparator', 'Relay', 'Switch'), blocks without a
    deterministic input to output map ('RNG', 'Counter'), and the discrete time
    blocks all fall into this category. For those a numerical Jacobian would
    produce a number that looks like a linearization but is not one, so they
    fail loudly instead.

    Blocks declare this through the 'linearizable' class attribute, or by
    raising from their 'to_statespace' method directly.

    Parameters
    ----------
    message : str
        optional message describing why linearization is not possible

    Example
    -------

    Catch it when linearizing a system that may contain switching blocks:

    .. code-block:: python

        from pathsim.exceptions import LinearizationError

        try:
            A, B, C, D = Sim.linearize()
        except LinearizationError as e:
            print(f"system is not linearizable: {e}")
    """
