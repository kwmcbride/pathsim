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


# HELPERS ===============================================================================

def _collect_leaves(val, out):
    """Recursively collect all leaf float values from a bus dict into *out*."""
    if isinstance(val, dict):
        for v in val.values():
            _collect_leaves(v, out)
    else:
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(0.0)


def _flatten_bus_to_array(d):
    """Return a 1-D float64 array of all leaf values in bus dict *d*."""
    leaves = []
    _collect_leaves(d, leaves)
    return np.array(leaves, dtype=float)


# CLASS =================================================================================

class ConnectionBooster:
    """Wraps a `Connection` instance and injects a fixed point accelerator. 

    This class is part of the solver structure and intended to improve the 
    algebraic loop solver of the simulation.

    Parameters
    ----------
    connection : Connection
        connection instance to be boosted with an algebraic loop accelerator

    Attributes
    ----------
    accelerator : Anderson
        internal fixed point accelerator instance
    history : float | int | array_like
        history, previous evaliation of the connection value
    """

    def __init__(self, connection):
        self.connection = connection
        self.history = self.get()

        # initialize optimizer (default args)
        self.accelerator = Anderson()

        # flat float64 snapshot of the previous bus-signal iteration (None for scalars)
        self._bus_history = None

        # tristate flag: None = not yet determined, True = bus, False = scalar
        self._is_bus = None


    def __bool__(self):
        return len(self.connections) > 0


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
        self._bus_history = None
        self._is_bus = None


    def update(self):
        """Wraps the `Connection.update` method for data transfer from source
        to targets and injects a solver step of the fixed point accelerator,
        updates the history required for the next solver step, returns the
        fixed point residual.

        For bus signals (object-dtype outputs carrying a dict) Anderson
        acceleration is not applicable.  The method falls back to plain
        fixed-point pass-through and measures convergence as the max
        absolute change in any leaf value across iterations.

        Returns
        -------
        res : float
            fixed point residual of internal fixed point accelerator
        """
        current = self.get()

        # Determine connection type on the first call after init/reset.
        if self._is_bus is None:
            self._is_bus = current.dtype == object and isinstance(current.flat[0], dict)

        if self._is_bus:
            # Bus signal: Anderson acceleration is not applicable.
            # Plain pass-through; convergence measured as max leaf-value change.
            new_flat = _flatten_bus_to_array(current.flat[0])
            if self._bus_history is None or self._bus_history.shape != new_flat.shape:
                res = float('inf')
            else:
                res = float(np.max(np.abs(new_flat - self._bus_history)))
            self._bus_history = new_flat   # snapshot before next in-place mutation
            self.set(current)
            return res

        # Normal scalar / array path — Anderson acceleration.
        _val, res = self.accelerator.step(self.history, current)
        self.set(_val)
        self.history = _val
        return res