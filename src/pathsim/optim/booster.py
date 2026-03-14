########################################################################################
##
##                               ConnectionBooster CLASS
##                                  (optim/booster.py)
##
##       class to boost connections, injecting a fixed point acelerator for loop
##             closing connections to simplify the algebraic loop solver
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from .anderson import Anderson


# CLASS =================================================================================

class ConnectionBooster:
    """Wraps a `Connection` instance and injects a fixed point accelerator.

    This class is part of the solver structure and intended to improve the
    algebraic loop solver of the simulation.

    For bus signals (object-dtype outputs carrying a flat float64 ndarray) Anderson
    acceleration operates directly on the flat buffer — no dict traversal needed.

    Parameters
    ----------
    connection : Connection
        connection instance to be boosted with an algebraic loop accelerator

    Attributes
    ----------
    accelerator : Anderson
        internal fixed point accelerator instance
    history : float | int | array_like
        history, previous evaluation of the connection value
    """

    def __init__(self, connection):
        self.connection = connection
        self.history = self.get()

        # initialize optimizer (default args)
        self.accelerator = Anderson()

        # bus-path state: None until first bus call
        self._bus_history = None
        self._anderson_out = None
        self._anderson_wrapper = None


    def __bool__(self):
        return True


    def get(self):
        """Return the output values of the source block that is referenced in
        the connection.

        Return
        ------
        out : float | int | array_like
            output values of source, referenced in connection
        """
        return self.connection.source.get_outputs()


    def set(self, val):
        """Set targets input values.

        Parameters
        ----------
        val : float | int | array_like
            input values to set at inputs of the targets, referenced by the
            connection

        """
        for trg in self.connection.targets:
            trg.set_inputs(val)


    def reset(self):
        """Reset the internal fixed point accelerator and update the history
        to the most recent value
        """
        self.accelerator.reset()
        self.history = self.get()
        # Clear bus history so the first post-reset call re-seeds (returns inf).
        self._bus_history = None


    def update(self):
        """Wraps the `Connection.update` method for data transfer from source
        to targets and injects a solver step of the fixed point accelerator,
        updates the history required for the next solver step, returns the
        fixed point residual.

        For bus signals (object-dtype outputs carrying a flat float64 ndarray)
        Anderson acceleration operates directly on the flat buffer.

        Returns
        -------
        res : float
            fixed point residual of internal fixed point accelerator
        """
        current = self.get()

        # Bus path: object-dtype register wrapping a flat float64 ndarray
        if current.dtype == object and len(current) > 0:
            bus_arr = current.flat[0]
            if isinstance(bus_arr, np.ndarray):
                if self._bus_history is None:
                    # First call after init/reset: seed history and pass through.
                    self._bus_history = bus_arr.copy()
                    self._anderson_out = bus_arr.copy()
                    self._anderson_wrapper = np.empty(1, dtype=object)
                    self._anderson_wrapper[0] = self._anderson_out
                    self.set(current)
                    return float('inf')
                # Anderson step on the flat buffer
                _val, res = self.accelerator.step(self._bus_history, bus_arr)
                np.copyto(self._bus_history, _val)
                np.copyto(self._anderson_out, _val)
                self.set(self._anderson_wrapper)
                return res

        # Scalar / array path — Anderson acceleration on register values
        _val, res = self.accelerator.step(self.history, current)
        self.set(_val)
        self.history = _val
        return res
